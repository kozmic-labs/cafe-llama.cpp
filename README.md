# cafe-llama.cpp

![ilustration](ilustration.png)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## MoE Offloading & Memory Optimization

In Mixture of Experts (MoE) models (such as **Qwen 3.8 Flash Next**, **DeepSeek-V2/V3**, **Mixtral**, etc.), expert weights represent the majority of parameters and VRAM. `cafe-llama.cpp` provides flags to offload MoE weights to **pinned host RAM (`CUDA_Host`)** or **CPU RAM** while keeping attention, KV cache, and routers on the GPU:

| Flag                  | Long Flag                | Description |
|-----------------------|--------------------------|---|
| `--pipeline-parallel` | `--no-pipeline-parallel` | Enable the offloading acceleration pipeline. The scheduler keeps 4 device copies of every streamed weight, so it is skipped (with a warning) when `4 x streamed weight bytes` does not fit in VRAM - that is the case for models with hundreds of experts per layer. |
| `-hmoe`               | `--host-moe`             | Keep **all MoE expert weights** in pinned host memory (`CUDA_Host_MoE`). On CUDA the GPU computes the experts itself: hot experts from a VRAM cache, the rest read straight from host RAM over PCIe. See "GPU-computed host experts" below. |
| `-nhmoe N`            | `--n-host-moe N`         | Same as `-hmoe` for the MoE weights of the **first N layers**. |
| `-cmoe`               | `--cpu-moe`              | Keep **all MoE expert weights** in CPU system RAM. |
| `-ncmoe N`            | `--n-cpu-moe N`          | Keep MoE weights of the **first N layers** in CPU system RAM. |
| `-ssd`                | `--ssd-streaming`, `--no-ssd-streaming` | Stream **routed expert weights from SSD on-demand** via `mmap`: only the experts a token actually selects page into RAM, the rest stay on disk. |
| `-nssd N`             | `--ssd-n-streaming N`    | Stream the MoE experts of the **first N layers** from SSD (implies `--ssd-streaming`); analogous to `-ncmoe`/`-nhmoe`, but the destination is SSD instead of RAM. |
| `-hmoed`              | `--host-moe-draft`       | Keep draft model MoE weights in pinned host memory (for speculative decoding). |
| `-nhmoed N`           | `--n-host-moe-draft N`   | Keep draft model MoE weights of the **first N layers** in pinned host memory (for speculative decoding). |
| `-cmoed`              | `--cpu-moe-draft`        | Keep draft model MoE weights in CPU system RAM (for speculative decoding). |
| `-ncmoed N`           | `--n-cpu-moe-draft N`    | Keep draft model MoE weights of the **first N layers** in CPU system RAM (for speculative decoding). |
| `--moe-cache MODE`   | `--no-moe-cache-profile` | Adaptively cache the hottest CPU-resident MoE experts in spare VRAM (`auto`, `on`, `off`, or `N` MiB). Computes resident expert hits on CUDA concurrently while CPU worker threads compute miss rows. |

### Streamed experts: chunk size, drafts and what is left on the table

When the routed experts are not in VRAM, every token pulls its own experts from host memory, so decode
speed is set by host bandwidth, not by the GPU. Two flags follow from that. Defaults stay as in upstream
llama.cpp, so set them yourself when you offload:

- **prompt chunk** `-b 8192 -ub 2048` (upstream default `2048 / 512`): one expert weight read serves the
  whole ubatch, so a bigger chunk reads the experts proportionally less often. Back off if the prompt
  buffers do not fit.
- **draft gating** `--spec-draft-p-min 0.5` (upstream default `0`): a draft token carries its own expert
  reads, so an unlikely draft costs more than it can win. Verification stays exact, only the wasted work
  goes away.
- **adaptive draft length** `--spec-adaptive` (default `off`): learn the real per-position acceptance from
  the verify results and keep only the drafts whose expected tokens beat their cost, down to skipping
  speculation entirely when even one draft does not pay. This is a portable version of Strata's draft
  controller; it needs no model change and applies to `draft-mtp`, `draft-eagle3`, `draft-dflash`,
  `draft-dspark` and `draft-simple` (the ngram types keep their own length policy). Tune the learning rate
  with `--spec-adaptive-decay` (EMA weight, default `0.1`). It is most useful on the offload path, where a
  rejected draft is a full set of expert reads.

Measured on an RTX 3090 (24 GB), Ryzen 7 5800X, 60 GB DDR4, `Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS`,
`-ngl 99 -nhmoe 34 -t 8 -fa on -ctk q8_0 -ctv q8_0`, same 4342-token prompt, 96 tokens generated, greedy:

| Flags | prompt t/s | decode t/s |
|---|---:|---:|
| `-b 1024 -ub 128 --pipeline-parallel -md mtp.gguf --spec-draft-n-max 3` (`--pipeline-parallel` is skipped, see the flag table) | 101 | 22.0 |
| `-b 2048 -ub 512 -md mtp.gguf --spec-draft-n-max 3 --spec-draft-p-min 0` | 58 | 14.7 |
| the same with `--spec-draft-p-min 0.5` | 242 | 22.5 |
| `-b 8192 -ub 2048 -md mtp.gguf --spec-draft-n-max 2 --spec-draft-p-min 0.5` | 421 | 23.2 |
| `-b 8192 -ub 2048`, no speculation | 477 | 24.6 |

What the numbers say: the chunk size is worth up to 4.7x on prompt processing, and an unfiltered MTP
window is worth less than nothing on this offload path (58 vs 242 prompt t/s, 14.7 vs 22.5 decode t/s).
Speculation only pays again once the drafts are filtered, but it stays a small loss at the largest
chunk (23.2 vs 24.6) - with the experts off, each extra token in the window is another full set of expert
reads, so a rejected draft is expensive here in a way it is not when the experts are in VRAM.

The expert read is the whole cost of decode, which shows in the thread count: same file, all experts in
pinned host memory (`-nhmoe 48`), decode t/s by `-t`: 1 -> 4.6, 4 -> 14.0, 8 -> 20.0, 16 -> 9.2. Use
physical cores, not SMT threads. `-t 16` loses because the kernel is bound by memory bandwidth, not by
cores, and SMT threads only fight over it.

What is left on the table: offload is **per layer**, so a streamed layer is read from host memory in full
every token, while routing is skewed enough that a cache holding a slice of *each* layer's experts serves
most lookups from VRAM for the same bytes. Same machine, same IQ3_XXS file, same request (1026 prompt
tokens, 96 generated, greedy):

| | prompt t/s | decode t/s (3 identical requests) |
|---|---:|---:|
| cafe-llama.cpp, `-b 8192 -ub 2048` and `--spec-draft-p-min 0.5` | 256 | 21.7, 21.9, 22.0 |
| Strata 0.1.24, same file, expert cache on | 256 | 45.3, 72.0, 96.4 |

Prompt processing is at parity at this size; the chunk size alone is what took cafe from 101 to 421 t/s on
the 4342-token prompt above (`-b`/`-ub` are the upstream defaults otherwise). The decode gap is the missing
tier, and it widens across identical requests because Strata keeps admitting the experts this conversation
routes to while cafe reads the streamed layers in full, every token. Closing it needs expert-granular placement: split each `ffn_*_exps` tensor into a
resident and a streamed part, remap the router ids to both, add the two results. It also needs the streamed
part to run while the GPU is busy - today the decode graph splits CUDA/CPU about 70 times per token and
every split ends in a synchronize, so the two halves wait on each other. Speculation stays close to a wash
for the same reason: a draft the target rejects is a full set of expert reads that bought nothing.

### GPU-computed host experts (`-hmoe`, CUDA)

With `-hmoe` / `-nhmoe` (and `-hmoed` / `-nhmoed` for the draft), the experts stay in pinned host memory, but the GPU runs the `MUL_MAT_ID` itself. Every expert tensor gets a table of per-expert base pointers in VRAM: a slot of the VRAM expert cache when the expert is resident, else its pinned host copy, which the kernel reads over PCIe. The router ids never leave the GPU, so a decode step is a single graph with no CPU/GPU synchronization, and it is captured as one CUDA graph (2 scheduler splits instead of ~70).

- The cache fills the free VRAM in the background on a low-priority stream, hottest experts first (use counts are read back from the device between steps), and replaces cold experts with hysteresis. It keeps 6% of VRAM free (1-3 GiB).
- Gate/up/SwiGLU are fused into one kernel. Large batches (prompt processing) gather the routed experts into VRAM and run the stock MMQ kernels.
- It works for every MoE model whose experts are `*_exps` tensors in a CUDA-supported quant type, not only Qwen4. `-cmoe` / `--moe-cache` keep the old CPU path.
- The expert heat is saved per expert tensor in `$LLAMA_CACHE/moe-direct` (else `~/.cache/llama.cpp/moe-direct`) every 1024 graphs and at exit. The next start fills the cache with those experts first, at up to 2 GiB per graph while VRAM is free, so the first request is not cold. The key includes the tensor bytes, so different models never share a file. This applies to the target, draft and MTP contexts alike.
- Half of the experts missing from VRAM are computed by CPU threads at the same time, from the same pinned RAM, so RAM bandwidth adds to the PCIe bandwidth (Strata's `pcie_frac`). The GPU picks those routes itself, writes their activations to mapped host memory and bumps a sequence number; the threads poll it, compute and post the results; a kernel later in the same graph waits for them and scatters them into the output. There is no driver call or stream sync between the halves, so the decode step stays one CUDA graph. Idle threads back off to a 50 us poll after 20 ms without work.
- Controls (environment): `GGML_CUDA_MOE_DIRECT=0` disables it, `GGML_CUDA_MOE_DIRECT_RESERVE_MB` sets the free VRAM to keep, `GGML_CUDA_MOE_DIRECT_BUDGET_MB` caps the cache, `GGML_CUDA_MOE_DIRECT_STATS=N` logs hit rate every N graphs, `GGML_CUDA_MOE_DIRECT_PROFILE=0` disables the saved heat (`GGML_CUDA_MOE_DIRECT_PROFILE_DIR` moves it), `GGML_CUDA_MOE_DIRECT_DECAY` sets the graphs per halving of the heat (default 128), `GGML_CUDA_MOE_DIRECT_CPU_FRAC` sets the CPU share of the misses (default 0.5, `0` disables the CPU threads), `GGML_CUDA_MOE_DIRECT_CPU_THREADS` their count (default: logical cores / 2 - 2).

Same machine and file, 1226-token prompt, 256 tokens, greedy, `-md mtp.gguf --spec-draft-n-max 3 --spec-draft-p-min 0.5 -ngl 99 -ngld 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -b 4096 -ub 1024 -t 8 --no-ngram`:

| | decode t/s (requests 1-6) | draft acceptance |
|---|---:|---:|
| `-cmoe --moe-cache on` | 21.4, 30.7, 30.5 | 83% |
| `-hmoe` (GPU-computed experts), cold start | 30.2, 59.2, 60.6, 62.1 | 83% |
| `-hmoe`, saved expert heat, misses on the GPU only (`CPU_FRAC=0`) | 36.5, 64.9, 63.2, 65.6, 65.9, 67.3 | 83% |
| `-hmoe`, saved expert heat, half of the misses on 6 CPU threads (default) | 52.1, 78.3, 78.7, 73.3, 81.1, 81.6 | 83% |

CPU share sweep (threads): 0.35 (4) 67-79, 0.5 (4) 66-80, 0.5 (6) 73-82, 0.7 (6) 74-79 t/s.

One target pass over a 4-token MTP window went from 98 ms to 25 ms (1 token: 34 -> 16 ms). Part of that is a CUDA fix that helps every model on Ampere: BF16/F16 matrices with few output rows (Qwen4 hyper-connection and router weights) now use the matrix-vector kernel for small batches instead of a one-block cuBLAS GEMM.

Where the decode time goes at steady state (~50 ms per verify step of the server): draft 3.3 ms, MTP process 0.4 ms, sampling/accept 1 ms, the rest is the target pass. With every expert in VRAM that pass costs 25 ms; with none it costs 218 ms (experts read at ~18 GB/s over PCIe 4.0). At the measured ~87% hit rate (12.8 GiB cache, 31% of the experts) the misses are the remaining ~20 ms, so more cache or fewer misses is what speeds this up further, not the speculative loop. Grouping the routes of a window by expert (one read per expert) was tried and was slower: the repeated reads hit L2, and more blocks in flight keep PCIe busier.


### CUDA MoE Expert Cache (`--moe-cache`)

To close the decode gap and bring Strata-style expert caching to `cafe-llama.cpp`, `--moe-cache` adaptively caches the hottest routed experts in spare VRAM when MoE weights are offloaded to host memory (`-hmoe`, `-nhmoe`, `-cmoe`, `-ncmoe`):

- **Modes**: `--moe-cache auto` (default), `--moe-cache on`, `--moe-cache off` (or `--moe-cache N` for an explicit budget in MiB).
- **Concurrent Execution**: When `MUL_MAT_ID` executes on the CPU, cached expert hits are dispatched asynchronously to CUDA on a dedicated stream while the CPU threads compute uncached misses in parallel, eliminating per-layer host-device synchronizations.
- **Universal MoE Support**: Applies to all MoE architectures in `cafe-llama.cpp` (Qwen4 / Qwen3.8-Flash-Next, DeepSeek-V2/V3/V4, Mixtral, OLMoE, Qwen2/3/3.5-MoE, Grok, Xing4, etc.).
- **Tuning & Prewarming**: `--moe-cache-profile` persists hot-expert heatmaps across sessions; `--moe-cache-expert-parallel` parallelizes across multiple GPUs. See `docs/backend/CUDA-MOE-CACHE.md` for complete technical details.


### Turbo KV Cache (low-bit K/V, GPU-only)

Low-bit **TurboQuant** types for the attention KV cache, selected with `-ctk` / `-ctv` (draft: `-ctkd` / `-ctvd`). Rows are stored in the rotated (WHT) domain and reconstructed inside the fused flash-attention kernel, so **flash-attention (`-fa`) is required**: it is auto-enabled, and loading aborts if flash-attention is explicitly disabled. Turbo KV types are GPU-side only.

| Type      | Approx bits/value | Compression vs fp16 | Constraint |
|-----------|-------------------|---------------------|------------|
| `turbo4`  | ~4.1              | ~3.9x               | none       |
| `turbo3`  | ~3.5              | ~4.6x               | head dim (K/V) must be a multiple of 128 |
| `turbo2`  | ~2.5              | ~6.4x               | head dim (K/V) must be a multiple of 128 |

Lower-bit K/V lets long-context KV fit in VRAM. Example: `-ctk turbo4 -ctv turbo4 -fa on`.

### Qwen 3.8 Flash Next / Qwen4 Internal N-Gram (PLE) Optimization

Qwen 3.8 Flash Next includes an internal Prompt Lookup Expert (PLE) N-gram hash embedding table (`per_layer_token_embd`, ~51B parameters). `cafe-llama.cpp` provides dedicated flags to manage its memory footprint:

| Flag          | Long Flag                               | Description |
|---------------|-----------------------------------------|---|
| `--ngram-ssd` | `--offload-ngram-ssd`, `--no-ngram-ssd` | Exclusively offload the internal N-gram embedding table to SSD on-demand via memory mapping (`mmap`), leaving active model layers in RAM/VRAM. |
| `--no-ngram`  | `--disable-ngram`, `--no-load-ngram`    | Force completely disable the internal N-gram embedding table and PLE layers, skipping all PLE tensors (0 bytes allocated in RAM/VRAM). |
| `--ngram`     | `--load-ngram`                          | Normal mode: load internal N-gram table and PLE layers into memory (default). |

Nothing in this family is on unless you ask for it: `-ssd`, `-nssd`, `--ngram-ssd`, `--ssd-direct`,
`--ssd-warm-dense`, `--ssd-release-mmap` and `--ssd-predict` all default to off. `-lzm` keeps the upstream
default (`auto`), which is what puts the N-gram table on the SSD and reads its rows per token: pass `-lzm off`
to keep those tensors resident when the machine has the RAM for it. The loader logs each marked tensor it
leaves on disk (`lazy read enabled`). `--ssd-predict` is off by default here, so `-ssd` streams
without predicting or hot-loading.

What the upstream `auto` default costs on this model, measured the same way as the table above (RTX 3090,
`-ngl 99 -nhmoe 34 -b 4096 -ub 1024`, no speculation, 4342-token prompt): with the N-gram table read from the
SSD 278 prompt / 17.8 decode t/s, with `--no-ngram` 357 / 23.0. Per-token row reads on the SSD are not free;
if the machine has the RAM for the table, `-lzm off` buys that back.

### Qwen 3.8 Flash Next & MTP Speculative Decoding

MTP (Multi-Token Prediction) draft models in GGUF format are available at:
👉 **[Hugging Face: quimmedes/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/quimmedes/Qwen3.8-Flash-Next-MTP-GGUF)**

Available quantizations:
- `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (~2.65 GB) - Recommended for balanced memory and speed
- `mtp-Qwen3.8-Flash-Next-Q6_K.gguf` (~3.24 GB)
- `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (~3.94 GB)
- `mtp-Qwen3.8-Flash-Next-BF16.gguf` (~7.40 GB) - Full precision


**Recommended command for Qwen 3.8 27B**
```sh
llama-server -m Qwen3.8-27B-Q5-v4-XYZ.gguf --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  -fa on -ctk q8_0 -ctv q8_0 -c 64000 -np 1 -t 8 -ctkd q4_0 -ctvd q4_0 -ngl 99 -ngld 99
  
  Agressive
  
 llama-server -m Qwen3.8-27B-Q5-v4-XYZ.gguf --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 \
  -fa on -ctk q8_0 -ctv q8_0 -c 64000 -np 1 -t 8 -ctkd turbo2 -ctvd turbo2 -ngl 99 -ngld 99 -lm mlock


```



**Recommended Server Command for Qwen 3.8 Flash Next:**
```sh

llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
-ctk q8_0 -ctv q8_0 -kvu \
 -fa on -ngl 99 -hmoe -c 64000 -np 1 --no-ngram 

Disable Ngram if you don't have enough RAM/VRAM
llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
-ctk q8_0 -ctv q8_0 -kvu \
 -fa on -ngl 99 -hmoe -c 64000 \
--no-ngram -np 1 --no-ngram 



MTP with offload

-b/-ub and --spec-draft-p-min are the two streamed-expert flags from the section above, spelled out here
because the upstream defaults (2048/512 and 0) leave a lot on the table.

llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -md mtp.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-p-min 0.75 \
  -ngl 99 \
  -hmoe \
  -fa on \
  -ctk q8_0 -ctv q8_0 -kvu \
  -ctkd q4_0 -ctvd q4_0 -ngld 99 \
  -c 64000 -b 2048 -ub 512 -np 1 \
  --no-ngram 
 
  
```

### Serving Safetensors Checkpoints

`-m` / `--model` accepts a Hugging Face safetensors checkpoint directly: either a directory with `config.json` and its shards, or a single `.safetensors` file. No GGUF file is produced - the checkpoint metadata is read into memory and the weights are streamed from the shards while the model buffers are filled, so the shards stay untouched and no extra disk space is used.

```sh
llama-server -m /path/to/Qwen3.6-35B-A3B-FP8 -c 32768
```

| Flag | Description |
|------|-------------|
| `--safetensors-outtype TYPE` | Storage type for the quantized weights of the checkpoint: `auto` (default) keeps FP8 weights in `q8_0` and packed int4 weights in `q4_0`, so neither expands to 16 bits. `f16`, `bf16`, `q8_0` and `q4_0` force one type for every quantized weight. NVFP4 weights are always kept in `nvfp4`, the flag does not apply to them. |
| `--safetensors-native` | Keep the FP8 weights of the checkpoint as they are stored, in the `f8_e4m3` type: the E4M3 codes plus the fp32 scale of every block of 128 values, 8.25 bits per weight. This is the same as `--safetensors-outtype native`, and it takes precedence over everything else the outtype asks for. |

Notes:

- `-mm` / `--mmproj` accepts the same checkpoint: the vision tower is built in memory as a CLIP model, so images work without a converted `mmproj-*.gguf`. The capabilities of the tower are read from the same in memory metadata, the checkpoint is never opened as a GGUF file.
- MTP: `--spec-type draft-mtp` uses the MTP layer that is part of the checkpoint, and `-md <checkpoint>` loads only its MTP block as a standalone draft (`-md` with a safetensors path is detected automatically). The block can sit in the main shards or in a shard of its own, like the `model_mtp.safetensors` of Qwen3.5.
- Weights stored as-is in the checkpoint (`BF16`, `F16`, `F32`) are read with `mmap` and take no extra RAM, exactly like a GGUF file. Everything else is produced by the source while the model buffers are filled: FP8 is dequantized with its 128x128 block scales or its per tensor scale, compressed-tensors int4 weights are unpacked with their per group scales, NVFP4 weights are repacked in the `nvfp4` blocks with their E4M3 scales, packed experts are collected into a single 3d tensor, norms and linear attention parameters are adjusted, and the vision tower is cast to F16/F32 - the same work the converter does, in memory. For a 35B FP8 checkpoint that is about 35 GB of RAM with `q8_0` (nothing is written to disk).
- `--safetensors-native` keeps FP8 weights in `f8_e4m3` instead of requantizing them to `q8_0`, so the codes and the block scales of the checkpoint survive bit for bit. The CPU vector kernels decode the codes directly, CUDA dequantizes them in the kernel and Vulkan in the shader. The type has no matrix-matrix kernels: on both GPU backends batches of up to 8 columns use the vector kernels, larger ones the dequantize `to_f16` path. CUDA offloads routed experts (`MUL_MAT_ID`) of MoE checkpoints as well, Vulkan does not, so on Vulkan the experts of a MoE checkpoint stay on the CPU. SYCL, Metal and WebGPU do not implement the type.
- The block of `f8_e4m3` covers 128 values and holds one fp32 scale, so the row length of every FP8 weight of the checkpoint must be a multiple of 128. The loader stops with an error naming the tensor otherwise.
- Exl3 checkpoints (ExLlamaV3) are decoded in memory as well: every quantized linear holds a trellis of 1 to 3 bit coded indices plus one factor per input and per output channel. The loader runs the reference decode (the 16 bit sliding window through the mul1 codebook, then the Hadamard transform of 128 along both axes with the factors) and emits a dense weight. The output type follows the bit width of each tensor - 1 and 2 bit tensors become `q2_k`, 3 bit tensors `q3_k` - so a 2.0 bpw 27B checkpoint lands near its own file size (about 12 GB resident for a 9.7 GB checkpoint). `--safetensors-outtype q8_0`, `f16`, `bf16` or `q4_0` overrides that with one higher precision type for every tensor when memory is not the constraint.
- The native path is verified against the same checkpoint loaded on the CPU: CUDA reproduces the CPU logits (`Mean KLD -0.000000`, `100 %` identical top tokens), Vulkan stays within its own rounding (`0.000108`, `99.2 %` identical top tokens).
- A modelopt `MIXED_PRECISION` checkpoint quantizes the attention layers to FP8 and the FFN and the LM head to NVFP4, layer by layer. The NVFP4 weights keep their original bits and their `weight_scale_2` and `input_scale` are loaded as the separate `<name>.scale` and `<name>.input_scale` tensors that the graph multiplies with, exactly like the converter stores them.
- The tokenizer is read from `vocab.json` and `merges.txt` when the checkpoint ships them, otherwise from `tokenizer.json`, so checkpoints that only carry the fast tokenizer load as well.
- The SSD streaming flags (`-ssd`, `-nssd`, `--ngram-ssd`) cannot page those tensors from disk, the loader logs a warning when they are requested.
- Supported checkpoints with a hand written mapping: `qwen3_5` (Qwen3.5 / Qwen3.8 dense), `qwen3_5_moe` (Qwen3.5 / Qwen3.6 MoE), `gemma4` / `gemma4_unified` (Gemma 4, text model of the unified multimodal checkpoints), `NemotronH` (Nemotron-H and Nemotron-3 Nano Omni, text model) and `agnes` (Agnes 3.0), text model and vision tower of the qwen and agnes families.
- The nemotron-h mapping reads the hybrid layout of that family: the layer pattern (`M` mamba2, `E` experts, `*` attention) decides the per layer feed forward length and kv head count, the mamba2 parameters keep the shapes this runtime expects (the group norm split by the ssm groups, A_log negated and exponentiated, D and A_log as columns), and the routed experts are NVFP4 packed per expert, stacked into one tensor with the per expert scales next to it. Its scale factors are the ones the converter writes: `context_length` of the hybrid window, the uniform head width for the rope, and the `pixtral` pre-tokenizer of the checkpoint.
- The gemma 4 mapping covers what that family needs on top of the plain decoder: the sliding window pattern per layer, the separate head dimensions and rope of the full attention layers, the shared kv layers, the per layer output scale, the key used as value where the checkpoint has no `v_proj`, and the frequency factors of the proportional rope, which the loader generates like the converter does. Its tokenizer is read as the sentencepiece style pieces of this repo (`gemma4` tokenizer), the token ids match the ones of the Hugging Face tokenizer.
- A checkpoint whose tensor names do not line up with what the architecture expects is refused with the name of the missing tensor, and a checkpoint that mixes several compressed-tensors groups (e.g. FP8 attention with NVFP4 FFN, as the unsloth NVFP4 releases do) is handled per tensor: the group of every tensor is resolved by name, and the global scales of the NVFP4 groups are applied as the reciprocal ggml expects.
- Any other checkpoint of a plain decoder is read by a generic mapping: the tensor names, the hyperparameters and the tokenizer come from `config.json` and the tokenizer files. The families known to it are llama, mistral and mixtral (`llama`), qwen2 and qwen2-moe, qwen3 and qwen3-moe, gemma, gemma2 and gemma3-text, olmo2, phi3 and starcoder2. It applies what those layouts need on top of the plain skeleton: the `(1 + w)` normalization of the gemma family, the q/k rotary permutation of the llama family, the attention biases of qwen2 and the routed experts of the MoE variants. This is pure data derived from the configuration, no per architecture code, and it is verified against the converter: loading such a checkpoint and the GGUF produced from it gives the same perplexity and logits.
- A checkpoint that needs more than that - fused or reshaped projections, a tokenizer that is not byte pair encoding (SentencePiece, WordPiece, tiktoken), or a non decoder layout - is refused with the reason instead of being loaded wrongly. Convert those to GGUF first.
- Packed int4 checkpoints (compressed-tensors `pack-quantized`, 4 bit weights, one scale per group of 128 columns) are loaded without expanding them to 8 or 16 bits. The routed experts of a packed checkpoint are not supported yet.
- NVFP4 weights are repacked from the modelopt nibble order into the ggml `nvfp4` blocks, which need a row length that is a multiple of 64. Packed NVFP4 experts are not supported yet.
- Agnes 3.0 runs a second FFN in parallel with the main one. It is loaded as separate tensors (`ffn_gate_par`, `ffn_up_par`, `ffn_down_par`) with the length in the new `*.feed_forward_parallel_length` key, and the graph adds its output to the main FFN - so every weight can be mapped from the checkpoint instead of being rewritten.
- Without `-nr` the fork copies the weights of CPU layers into repacked host buffers; add `-nr` to keep them memory mapped when serving a checkpoint that does not fit in RAM, and `-ngl N` to place part of the layers in VRAM.


## Building from Source

### 1. NVIDIA CUDA (Windows / Linux)
```sh
# CMake configure with CUDA backend
cmake -B build -DGGML_CUDA=ON

# Build Release
cmake --build build --config Release -j 2 
```

### 2. Vulkan (Cross-Platform AMD / Intel / NVIDIA)
```sh
# Requires Vulkan SDK installed
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j 2
```

### 3. AMD ROCm / HIP (Linux / Windows)
```sh
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1100;gfx1030"
cmake --build build --config Release -j 2
```

### 4. Apple Metal (macOS)
```sh
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j 2
```

### 5. CPU Only (AVX2 / AVX-512)
```sh
cmake -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build --config Release -j 2
```


-j N : The number of CPU threads the processor will use to compile, 
a safe number is the amount of physical cores of the processor.

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity


The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
