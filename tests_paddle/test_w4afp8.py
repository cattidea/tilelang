import paddle
paddle.enable_compat(scope={"tilelang"})

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tvm import tirx

import numpy as np
import random
import unittest

import math
from functools import partial

import itertools

block_M = [256]
block_N = [64, 128, 256]
block_K = [128]
num_stages = [2, 3]
threads = [128, 256]
enable_rasterization = [True, False]
_configs = list(
    itertools.product(
        block_M,
        block_N,
        block_K,
        num_stages,
        threads,
        enable_rasterization,
    )
)

configs = [
    {
        "block_M": c[0],
        "block_N": c[1],
        "block_K": c[2],
        "num_stages": c[3],
        "threads": c[4],
    } for c in _configs
]

def _tir_u8_to_i4_to_f8(nbit: int, val: tirx.PrimExpr, pos: tirx.PrimExpr):
    assert nbit == 4
    assert val.dtype == "uint8"

    high = val << (pos.astype("uint8") * tirx.const(nbit, "uint8"))
    sign = (tirx.const(1, "uint8") << tirx.const(7, "uint8")) & high

    mask = tirx.const(7, "uint8") << tirx.const(4, "uint8")
    i4 = high & mask

    i8 = (i4 >> tirx.const(4, "int8")) | sign
    return tirx.reinterpret("float8_e4m3", i8)

@autotune(
    configs=configs,
    warmup=3,
    rep=20,
)
@tilelang.jit(out_idx=[3])
def w4afp8_gemm(tokens, M, N, K, Batch, block_M=256, block_N=128, block_K=128, num_stages=2,
                threads=256, in_dtype="float8_e4m3", out_dtype="bfloat16",accum_dtype="float16"):
    num_elems_per_byte = 2
    storage_dtype = "uint8"
    W_shared_shape = (block_N, block_K // num_elems_per_byte)
    W_dequantize_local_shape = (block_N, block_K)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in tokens)
    K_Group = K // 128
    
    @T.prim_func
    def main(
            X: T.Tensor((M, K), in_dtype),
            W: T.Tensor((Batch, N, K // num_elems_per_byte), storage_dtype),
            W_S: T.Tensor((Batch, K_Group, N), "float32"),
            Ct: T.Tensor((M, N), out_dtype),
            batch_sizes: T.Tensor([Batch], "int32"),  # type: ignore
            batch_offsets: T.Tensor([Batch], "int32"),  # type: ignore
            batch_padded_offsets: T.Tensor([Batch], "int32"),  # type: ignore
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), total_m_blocks, threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), in_dtype)
            W_shared = T.alloc_shared(W_shared_shape, storage_dtype)
            WS_shared = T.alloc_shared((block_N), "float32")
            W_dequantize_local = T.alloc_fragment(W_dequantize_local_shape, in_dtype)
            Ct_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)


            cur_batch_idx = T.alloc_local([1], "int32")
            cur_batch_size = T.alloc_local([1], "int32")
            m_start_padded = by * block_M
            for i in range(Batch):
                in_cur_batch_idx = (m_start_padded >= batch_padded_offsets[i])
                cur_batch_idx[0] = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx[0])
            cur_batch_size[0] = batch_sizes[cur_batch_idx[0]]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx[0]] + batch_offsets[
                cur_batch_idx[0]]
            
            T.clear(Ct_local_accum)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(W[cur_batch_idx[0], bx * block_N :(bx + 1) * block_N, k * block_K // num_elems_per_byte : (k + 1) * block_K // num_elems_per_byte], W_shared)
                
                for i, j in T.Parallel(block_N, block_K):
                    W_dequantize_local[i, j] = _tir_u8_to_i4_to_f8(
                        4,
                        W_shared[i, j // num_elems_per_byte],
                        j % num_elems_per_byte
                    )
                T.copy(X[m_start:m_start + block_M, k * block_K:(k + 1) * block_K], X_shared)

                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                T.clear(Ct_local)
                T.gemm(W_dequantize_local, X_shared, Ct_local, transpose_B=True)
                T.copy(W_S[cur_batch_idx[0], k : (k + 1), bx * block_N : (bx + 1) * block_N], WS_shared)
                for i, j in T.Parallel(block_N, block_M):
                    Ct_local_accum[j, i] += Ct_local[i, j] * T.cast(WS_shared[i], 'float16')

            for i, j in T.Parallel(block_M, block_N):
                with T.If(i < cur_batch_size[0]), T.Then():
                    Ct[m_start + i, bx * block_N + j] = Ct_local_accum[i, j]

    return main


def construct_inputs(tokens, padding_M):
    batch_count = len(tokens)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + tokens[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((tokens[i]) / padding_M) * padding_M)
    tokens = paddle.to_tensor(tokens, dtype="int32")
    batch_offsets = paddle.to_tensor(batch_offsets_list, dtype="int32")
    batch_padded_offsets = paddle.to_tensor(batch_padded_offsets_list, dtype="int32")
    return tokens, batch_offsets, batch_padded_offsets

def w4afp8_gemm_complie(input_fp8, weight_int4, weight_dequant_scale, tokens):
    block_M = 256
    padding_M = block_M
    M = sum(tokens)
    batch_count = len(tokens)

    N = weight_int4.shape[1]
    K = input_fp8.shape[1]
    kernel = w4afp8_gemm(tuple(tokens),
        M, N, K, batch_count)
    print(kernel.get_kernel_source())

    batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(tokens, padding_M)
    def run():
        return kernel(input_fp8, weight_int4, weight_dequant_scale, batch_sizes, batch_offsets, batch_padded_offsets)
    return run

class TestTLW4AFP8GEMM(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.test_cases = [
            # [1, 2, 512, 256],
            # [16, 2, 512, 256],
            # [128, 2, 512, 256],
            # [256, 2, 512, 256],
            # [512, 2, 512, 256],
            # [1024, 2, 512, 256],
            # [1, 2, 8192, 1024],
            # [16, 2, 8192, 1024],
            # [128, 2, 8192, 1024],
            # [512, 2, 8192, 1024],
            # [128, 16, 8192, 1024],
            # [512, 16, 8192, 1024],
            # [64, 8, 7168, 8192],
            # [128, 8, 7168, 8192],
            # [256, 8, 7168, 8192],
            # [512, 8, 7168, 8192],
            # [1024, 8, 7168, 8192],
            [2048, 8, 7168, 8192],
            # [4096, 8, 7168, 8192],
        ]

    def set_data(self, tokens_per_group, Experts, N, K):
        paddle.seed(0)
        self.tokens_per_group = tokens_per_group
        self.N = N
        self.K = K
        self.BATCH = Experts
        self.TokenPadding = 0

        tokens = [self.tokens_per_group] * self.BATCH
        self.tokens_prefix_sum = np.cumsum(tokens)
        self.tokens = paddle.to_tensor(tokens, dtype="int64")
        self.all_tokens = int(self.tokens.sum())
        self.tokens_prefix_sum = paddle.to_tensor(self.tokens_prefix_sum, dtype="int64")

        # Prepare input data
        self.input_fp8 = paddle.randn([self.all_tokens, self.K], dtype="bfloat16").astype(paddle.float8_e4m3fn)
        self.weight = paddle.randn([self.BATCH, self.N, self.K], dtype="bfloat16")
        # Prepare scales
        self.weight_scale = 7 / self.weight.abs().max(axis=-1).reshape([self.BATCH, self.N, 1])
        self.weight_quant = (self.weight * self.weight_scale.reshape([self.BATCH, self.N, 1])).astype("int")
        self.weight_quant = paddle.clip(self.weight_quant, -7, 7)
        self.weight_quant = self.weight_quant.astype("bfloat16")
        self.weight_quant = paddle.where(self.weight_quant > 0, self.weight_quant, 8 - self.weight_quant)
        self.weight_dequant_scale = 1 / self.weight_scale.astype("float32")
        
    
    def test_tl_w4afp8_gemm(self):
        for [tokens_per_group, Experts, N, K] in self.test_cases:
            self.set_data(tokens_per_group, Experts, N, K)
            # Run TileLang implementation
            tokens = [self.tokens_per_group] * self.BATCH
            weight_dequant_scale = (self.weight_dequant_scale * 512).repeat_interleave(self.K // 128, axis=-1).transpose([0, 2, 1]).contiguous()
            def get_weight_int4(weight_quant):
                uint8_w = weight_quant.astype("uint8")
                ui4_left = uint8_w[..., 0::2] << 4
                ui4_right = uint8_w[..., 1::2]
                weight_int4 = ui4_left | ui4_right
                return weight_int4

            run_kernel = w4afp8_gemm_complie(self.input_fp8,
                get_weight_int4(self.weight_quant),
                weight_dequant_scale,
                tokens)
            out = run_kernel()
            print(out)

if __name__ == "__main__":
    unittest.main()
