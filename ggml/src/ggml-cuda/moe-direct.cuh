#pragma once

#include "common.cuh"

// MoE experts in pinned host memory (CUDA_Host), computed on the GPU.
// Each expert tensor gets a device table of per-expert base pointers: a VRAM slot when the expert is cached,
// else the pinned host copy (read over PCIe). The routing never leaves the GPU, so a decode step has no
// host round trip and can run as one CUDA graph. The host fills VRAM slots between graphs from device-side use counts.

struct ggml_cuda_moe_direct_ctx;

bool ggml_cuda_moe_direct_enabled();

// src0 must already be known to be in a CUDA_Host buffer
bool ggml_cuda_moe_direct_type_supported(const ggml_tensor * src0);

// Registers new expert tensors of the graph, applies finished fills and schedules new ones.
// Must run before graph capture or evaluation starts.
void ggml_cuda_moe_direct_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph,
        bool (*is_direct)(const ggml_tensor * src0));

// Computes dst = MUL_MAT_ID(src0, src1, ids). When gate_src0 is set, computes up * silu(gate) into out instead.
// Returns false when the tensor is not registered; the caller then uses the stock path.
bool ggml_cuda_moe_direct_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * gate_src0, ggml_tensor * out,
        void (*mul_mat_id)(ggml_backend_cuda_context & ctx, ggml_tensor * dst));

// Largest token count that uses the pointer-table matvec; larger batches gather the experts into a dense tensor.
int ggml_cuda_moe_direct_mmv_max_tokens();

void ggml_cuda_moe_direct_free(ggml_cuda_moe_direct_ctx * md);
