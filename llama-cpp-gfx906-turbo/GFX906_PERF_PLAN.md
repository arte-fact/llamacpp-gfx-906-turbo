# Plan: gfx906 Performance Improvements — Evaluation & Implementation

## Context

10 performance improvement proposals were evaluated for the gfx906 (4×MI50) llama.cpp fork. After thorough code exploration, **6 are already done or not actionable**, and **4 are worth implementing**. The fundamental bottleneck is kernel launch latency (720 launches/token, ~3.6ms), not bandwidth (10% utilization).

Current perf: 57 tok/s (f16 KV), 46 tok/s (turbo3 KV) on Qwen3-Coder-Next 80B Q4_1.

## Already Done / Not Actionable (no work needed)

| # | Proposal | Status |
|---|----------|--------|
| 1 | Shadow Cache V Dequant | **DONE** — NEXT_OPTIMIZATIONS.md: "Shadow cache with stride fix: best turbo3 path". The 18% is inherent turbo3 overhead, not a bug. |
| 3 | DPP Softmax | **DONE** — FA vec kernel uses pure DPP reductions (xor1/2/4/8/16 in `gfx906-common.cuh`). No LDS. |
| 6 | HIP Graphs | **DONE** — `-DGGML_HIP_GRAPHS=ON`, +8-10% gen speed. |
| 7 | s_waitcnt Tuning | **NOT ACTIONABLE** — Only 3 instances, all correct (lgkmcnt for LDS, vmcnt for global). |
| 8 | Overclock | **NOT ACTIONABLE** — "At batch=1, power is NOT the bottleneck — latency is." (NEXT_OPTIMIZATIONS.md) |
| 9 | Speculative Decoding | **DONE** — ngram-mod gives +47% on warm cache. SSM-hybrid models can't use it. |
| 10 | Qwen3-Coder-Next Hang | **NOT CONFIRMED** — No hang evidence. FWHT enforces `n_elements % 128 == 0`. The 18% turbo3 cost is expected. |

---

## Implementation Plan (4 actionable items)

### Phase 1: FWHT Register-Based Rewrite (HIGH impact, LOW risk)

**Problem**: `kernel_turbo_wht` and `kernel_set_rows_turbo3` both run at `__launch_bounds__(1, 1)` — 1 thread per block — wasting 63/64 Wave64 lanes. This was a workaround for a HIP compiler bug with shared-memory FWHT.

**Key insight**: `kernel_set_rows_turbo4` already proves the fix — it runs at `__launch_bounds__(256, 1)` using `turbo_fwht_128(x)` on a local `float x[128]` register array. The HIP compiler bug only affects shared-memory FWHT, not register-based FWHT.

**Changes**:

1. **`kernel_set_rows_turbo3`** (turbo-quant.cu:523-630): Rewrite to match turbo4 pattern:
   - Change `__launch_bounds__(1, 1)` → `__launch_bounds__(256, 1)`
   - Replace `__shared__ float x[128]` → local `float x[128]`
   - Use `grp_idx = blockIdx.y * blockDim.x + threadIdx.x` (like turbo4:700)
   - Call `turbo_fwht_128(x)` instead of inline butterfly
   - Update dispatch (turbo-quant.cu:658-665): `dim3 grid(ne01, (n_groups_per_row + 255) / 256)` + `dim3 block(256)`

2. **`kernel_turbo_wht`** (turbo-quant.cu:472-509): Same rewrite:
   - Change `__launch_bounds__(1, 1)` → `__launch_bounds__(256, 1)`
   - Replace `__shared__ float x[128]` → local `float x[128]`
   - Each thread processes one 128-element group independently
   - Load signs from `__constant__`, apply `turbo_fwht_128(x)`, scale, store
   - Update dispatch (turbo-quant.cu:968): `<<<(n_groups + 255) / 256, 256>>>`

**Files**: `ggml/src/ggml-cuda/turbo-quant.cu` (lines 472-509, 523-666, 953-970)

**Verify**: Build → run turbo3 inference → compare output quality and tok/s vs baseline 46 tok/s.

---

### Phase 2: Packed FP16 Shadow Dequant (LOW-MEDIUM impact, LOW risk)

**Problem**: The 3 shadow dequant kernels (fattn.cu:18-119) use scalar `__half2float` → float multiply → `__float2half` per element. gfx906 has `v_pk_mul_f16` that does 2x FP16 values per VALU cycle.

**Changes**: For each of the 3 dequant kernels (`k_turbo3_dequant_rows_f16`, `k_turbo2_dequant_rows_f16`, `k_turbo4_dequant_rows_f16`):

1. Process 2 elements per loop iteration (j += blockDim.x * 2, or process pairs)
2. Pack norm into `half2`: `__half2half2(blk->norm, blk->norm)`
3. Look up 2 codebook values, pack into `half2`
4. Use `__hmul2(codebook_pair, norm2)` for the multiply
5. Store result as `half2` (aligned 4-byte write)
6. Handle odd-element tail if ne0 is odd (unlikely but safe)

**Files**: `ggml/src/ggml-cuda/fattn.cu` (lines 18-119)

**Verify**: Bit-exact comparison of dequant output → turbo3 tok/s comparison.

---

### Phase 3: MMVQ Software Pipelining (MEDIUM impact, MEDIUM risk)

**Problem**: gfx906 MMVQ kernels (`mmvq-q4_0.cuh` etc.) serialize load → compute → load → compute. Global loads have ~300-400 cycle latency. At 24-32 VGPRs per thread with 256 available, there's room for double-buffering.

**Changes**: Add L2 prefetch to MMVQ kernels, reusing the pattern from `mmq-prefetch.cuh`:

1. Before main loop: issue `global_load_dword` asm for first iteration's data
2. Each iteration: prefetch next iteration's data, consume current via VALU
3. Keep `__launch_bounds__(64, 1)` — single wavefront with ~50-60 VGPRs is fine

**Files**:
- `ggml/src/ggml-cuda/gfx906/matmul/mmvq-q4_0.cuh`
- `ggml/src/ggml-cuda/gfx906/matmul/mmvq-q4_1.cuh`
- `ggml/src/ggml-cuda/gfx906/matmul/mmvq-q8_0.cuh`

**Verify**: `rocm-objdump` for VGPR count (ensure no spills) → `rocprof` kernel timing → overall tok/s.

---

### Phase 4: Manual P2P Ring-Reduce for TP (HIGH impact, HIGH risk)

**Problem**: Pipeline parallelism causes ~8ms bubble (5 graph splits, 4 GPUs). True tensor parallelism with allreduce would eliminate this.

**Scope**: This is a multi-week effort touching the entire stack:
- New ring allreduce kernel over `hipMemcpyPeerAsync` P2P
- New `GGML_OP_ALLREDUCE` in the backend
- Graph construction changes for row-split matmuls
- Split-mode row currently crashes on HIP — root cause must be found first

**Recommendation**: Defer until Phases 1-3 are validated. Consider a simpler intermediate step: overlap P2P transfers with compute in the existing PP path using CUDA streams.

**Files**: New `gfx906/comm/ring-allreduce.cuh`, modifications to `ggml-cuda.cu`, `llama-model.cpp`, `ggml.c`

---

## Priority Summary

| Phase | Item | Effort | Expected Gain | Risk |
|-------|------|--------|---------------|------|
| 1 | FWHT register rewrite | 1-2 hours | 1-5% turbo3 speed | Low |
| 2 | Packed FP16 dequant | 2-4 hours | 0.5-2% turbo3 speed | Low |
| 3 | MMVQ prefetch | 1-2 days | 1-3% all models | Medium |
| 4 | P2P ring allreduce | 2-4 weeks | 10-20% all models | High |

Phases 1+2 can be done together. Phase 3 is independent. Phase 4 should wait.
