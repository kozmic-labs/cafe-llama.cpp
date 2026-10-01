#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Runs one batched matvec over slot-pool experts selected by device-side IDs.
// Output row c is W[ids[c]] times activation row act_ids[c].
// slot_stride_bytes is the original tensor nb[2], and act_q8 uses the padded q8_1 layout from quantize_row_q8_1_cuda.
// force_dedicated selects the MoE kernel even when modulo routing needs no explicit activation map.
enum class ggml_cuda_moe_cache_mmv_path {
    generic,
    dedicated,
};

ggml_cuda_moe_cache_mmv_path ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    bool force_dedicated, cudaStream_t stream);

// Runs up * GLU(gate) for rows whose two independently cached experts are resident.
void ggml_cuda_moe_cache_mmv_fused(
    const void * up_pool, const void * gate_pool, ggml_type type0,
    const char * act_q8, const int32_t * up_ids_dev,
    const int32_t * gate_ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    float up_min, float up_max, float gate_min, float gate_max,
    cudaStream_t stream);

// Matvec for MUL_MAT_ID where each expert base address comes from table[e] (VRAM slot or pinned host memory).
// vy is src1 in the padded q8_1 layout of quantize_row_q8_1_cuda. When gate_table is set, dst = up * silu(gate).
// counts[e] (and gate_counts[e]) are incremented once per routed row.
struct ggml_cuda_moe_direct_mmv_args {
    const uint64_t * table;
    const uint64_t * gate_table;
    const void     * vy;
    const int32_t  * ids;
    float          * dst;
    uint32_t       * counts;
    uint32_t       * gate_counts;
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t stride_row_x;
    uint32_t n_used;
    uint32_t n_routes;
    uint32_t ids_stride;
    uint32_t nchannels_y;
    uint32_t stride_channel_y;
    uint32_t stride_col_y;
    uint32_t stride_channel_dst;
    uint32_t stride_col_dst;
    const uint8_t * skip; // optional, per route: 1 = computed elsewhere (CPU), write nothing
};

bool ggml_cuda_moe_direct_mmv_supported(ggml_type type);
void ggml_cuda_moe_direct_mmv(ggml_type type, const ggml_cuda_moe_direct_mmv_args & args, cudaStream_t stream);
