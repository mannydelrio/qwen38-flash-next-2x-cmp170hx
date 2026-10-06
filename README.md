# Qwen3.8-Flash-Next on 2× CMP 170HX

## TL;DR

| | |
|---|---|
| **5.7× faster** | on a 100K-token prompt |
| **14.4 s** | 100K-token prompt + 256-token answer (was 81.4 s) |
| **109–159 tok/s** | per answer, prose to code (was ~37) |
| **262K** | token context, full native length |

- **What:** Qwen3.8-Flash-Next-Uncensored, a 512-expert MoE in 4-bit AWQ, served by vLLM on two NVIDIA CMP 170HX
  mining cards (unlocked to 64 GB each) in a 16 GB-RAM desktop.
- **How:** the model is split across the two cards in a pipeline, a built-in draft head predicts 3 tokens ahead, and
  the 29.8 GB embedding table is read from the NVMe SSD while the GPUs work, which frees enough GPU memory for the
  full context.
- **Quality:** unchanged by the speedups: HumanEval 155/164, 8/8 on a long-document test up to 222K tokens, and the
  same predictions as a reference run within normal run-to-run noise.
- **Try it:** follow [Setup](#setup) below; everything it needs is in this repo.

## Overview

A measured, reproducible setup for running **Qwen3.8-Flash-Next-Uncensored** (512-expert MoE, AWQ W4A16) on two
NVIDIA CMP 170HX mining cards with vLLM: **4–5.7× faster** than our starting point at every prompt size, with the
model's full **262,144-token** context, on a 16 GB-RAM desktop.

The main trick: the model's 29.8 GB per-layer embedding (PLE) table is served **from the NVMe SSD** instead of GPU
memory, with reads overlapped with GPU work. That frees enough VRAM to run speculative decoding and the full
context at the same time.

> **Experimental.** This uses a patched, unsigned NVIDIA kernel driver ([cmpunlocker](https://github.com/amoghmunikote/cmpunlocker))
> and a community vLLM fork. All numbers below come from our two cards. Check the model's license
> (`qwen-community-1.0`) before use; the uncensored model has had its refusal behavior removed.

## Results

Time to a 256-token answer (prompt processing plus generation), average of two isolated runs, same fixed prompts.
"Before" is a stock tensor-parallel vLLM setup on the same cards.

| Prompt | Before | Now | Faster |
|---|---:|---:|---:|
| 1K tokens | 7.5 s | 1.7 s | 4.4× |
| 8K tokens | 13.0 s | 2.9 s | 4.5× |
| 32K tokens | 31.2 s | 6.0 s | 5.2× |
| 64K tokens | 54.5 s | 9.7 s | 5.6× |
| 100K tokens | 81.4 s | 14.4 s | 5.7× |
| 4 requests at once, 8K each | 31.0 s | 7.8 s | 4.0× |

| Metric | Before | Now |
|---|---:|---:|
| Answer generation, 1 request (prose / code) | ~37 tok/s | 109 / 159 tok/s |
| Answer generation, 4 at once (total, prose / code) | – | 283 / 416 tok/s |
| Prompt reading | ~1,340 tok/s | ~7,700 tok/s |
| First token, 100K prompt | 74.6 s | 13.0 s |
| Max context per request | 258K | 262K |
| KV cache (total) | 260K tokens | 897K tokens |
| HumanEval (164, thinking off, temp 0.6) | – | 155/164 (runs range 154–158) |
| Long-document test (8 questions, up to 222K tokens) | – | 8/8 |

## Hardware and software we used

| Part | Version |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX, 8 GB variant (PCI ID `10de:20c2`), unlocked to 64 GB each |
| CPU / RAM | AMD Ryzen 5 5600X, 16 GB RAM |
| Disk | 1 TB NVMe (Crucial P2). The converted model is 110 GB |
| PCIe | Gen2 x4 per card (stock board, no capacitor mod, no P2P) |
| OS | Ubuntu 26.04, kernel 7.0.0-29 |
| GPU driver | [nvidia-open 610.43.02](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.43.02) + [cmpunlocker v0.5](https://github.com/amoghmunikote/cmpunlocker/releases/tag/v0.5) |
| Inference engine | [wtdcode/vllm-backport @ `3142ded1`](https://github.com/wtdcode/vllm-backport/tree/3142ded1671b9565503975ee946d7e19fe50ea95) + [`patches/`](patches/) (PyTorch 2.13.0, Triton 3.7.1, CUDA 13.2 in its venv) |
| Model | [leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32 @ `fa56146`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32/tree/fa56146238f9fcd5ab591b7052b31e28efdca5c3), an AWQ build of [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) |

## Repository layout

```
patches/vllm-backport-3142ded1.patch   pipeline-parallel KV fix (from clee/vllm-sm80) + vLLM PR #57105
plugins/tinybeast-ple-int4/            vLLM plugin: 4-bit PLE table format and lookup kernel
plugins/tinybeast-ple-ssd/             vLLM plugin: serve the PLE table from the SSD, CPU-side prefetch, GPU memory cap
scripts/prepare_checkpoint.py          download the checkpoint (resumable) and convert the PLE table to 4-bit
scripts/checkpoint-source.json         which checkpoint and revision to download
serve/serve.sh                         the exact production launcher
serve/tinybeast-vllm.service           optional systemd user unit
tests/                                 the test scripts behind the numbers above
```

Everything expects one root folder (default `/opt/flashnext`, override with `FLASHNEXT_ROOT`):

```
$FLASHNEXT_ROOT/
  vllm-backport/          the vLLM fork, patched, with its .venv
  model-awq-int4ple/      written by prepare_checkpoint.py
  download-staging/       temporary, used during conversion
  checkpoint-source.json  copy of scripts/checkpoint-source.json
```

## Setup

1. **Unlock the cards.** Install nvidia-open 610.43.02, then cmpunlocker **v0.5 or newer**:

   ```bash
   sudo ./install.sh --profile=8gb --no-iommu
   ```

   If you don't pass the cards to VMs, disable the passthrough helper (it blocks GPU resets):

   ```bash
   sudo systemctl disable cmpunlocker-passthrough.service
   ```

   Then fully power off and back on. `nvidia-smi` should show 65536 MiB per card and ECC Enabled.
   Older cmpunlocker builds (v0.2) hand out a protected 141 MB region at the top of memory, and long prompts then
   crash the second card with Xid 31.

2. **Build vLLM.** Clone the fork into `$FLASHNEXT_ROOT/vllm-backport`, check out `3142ded1671b9565503975ee946d7e19fe50ea95`,
   apply the patch, and build it into `.venv` per that repo's instructions:

   ```bash
   git apply /path/to/this/repo/patches/vllm-backport-3142ded1.patch
   ```

3. **Get the model.** Copy `scripts/checkpoint-source.json` into `$FLASHNEXT_ROOT`, then run the conversion with the
   vLLM venv's Python (needs `requests`, `torch`, `huggingface_hub`, `safetensors`):

   ```bash
   FLASHNEXT_ROOT=/opt/flashnext python scripts/prepare_checkpoint.py
   ```

   It downloads shard by shard (resumable), never keeps two full checkpoints, and prints `READY` when done.

4. **Install the plugins** into the vLLM venv:

   ```bash
   $FLASHNEXT_ROOT/vllm-backport/.venv/bin/python -m pip install -e plugins/tinybeast-ple-int4 -e plugins/tinybeast-ple-ssd
   ```

5. **Launch.** Copy `serve/serve.sh` into `$FLASHNEXT_ROOT` and run it (or use the systemd unit). The first start takes
   about 5.5 minutes; it serves an OpenAI-compatible API on `127.0.0.1:18084` as `tinybeast-flashnext-uncensored`.

6. **Check your numbers** with the scripts in `tests/` (they read the API key, if any, from `VLLM_API_KEY`):

   ```bash
   python3 tests/needle.py 200000 0.5
   ```

   - `needle.py TOKENS DEPTH`: hides a code in a long prompt and asks for it back.
   - `lc_hard.py LABEL 64000 128000 245000`: the 8-question long-document test (set `HAYSTACK_DIR` to a folder of `.py` files).
   - `decode_speed.py`: answer speed for prose and code, 1 and 4 requests at once.
   - `tf_compare.py dump OUT.json` / `compare REF.json NEW.json`: teacher-forced quality comparison between two configs.

## Settings that matter

| Setting | Value | Why |
|---|---|---|
| Parallelism | `--pipeline-parallel-size 2`, `VLLM_PP_LAYER_PARTITION=25,23` | No P2P on these cards, so tensor parallelism pays a host-memory sync on every layer; 25/23 balances the two stages |
| Speculative decoding | MTP, 3 draft tokens | About 2.5× faster answers; output is unchanged |
| PLE table | `VLLM_PLUGINS=tinybeast_ple_int4,tinybeast_ple_ssd`, `TB_PLE_SSD=1`, `TB_PLE_SSD_PREFETCH=1` | Frees ~29 GB of VRAM; prefetch computes row ids on the CPU for prompt steps so SSD reads overlap the previous GPU step |
| Prompt chunk | `--max-num-batched-tokens 4096` | With prefetch: 100K prompt 28.0 s → 14.6 s |
| GPU memory | `--gpu-memory-utilization 0.94`, `TB_TORCH_MEM_CAP_GIB=61.5`, `CUDA_MODULE_LOADING=EAGER` | vLLM under-measures the second pipeline stage's working memory; this keeps headroom on that card |
| MoE kernel | `VLLM_DETERMINISTIC_MOE_ALIGN=0` | About 17% faster answers, same quality |
| Power | 180 W per card | 220 W / 250 W gave only 5.6% / 8.4% faster long prompts for 19% / 36% more GPU power |

Optional plugin switches: `TB_PLE_SSD_VERIFY=N` checks the first N CPU-computed row ids against the GPU's (falls back to
the GPU path on any mismatch); `TB_PLE_SSD_TIMING=1` logs GPU-wait vs SSD-read time per prompt step.

## What we tried and dropped

- **Context past 262K (YaRN ×2 → 524K):** works (8/8 on the long-document test at 441K tokens, hidden code found at
  502K) but shifts outputs slightly at every length (mean KL 0.019 vs a 0.011 run-to-run floor). Our bots never used
  more than 191K, so we stay native.
- **Higher power limits:** see the table above.
- **vLLM PRs #58114 and #58449:** no gain / startup crash on this fork.
- **80 GB of VRAM:** not possible on the 8 GB variant (4 HBM stacks × 16 GB = 64 GB). **PCIe Gen3:** unsolved anywhere (fused off).

## Built on

- [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker): the CMP 170HX memory and compute unlock. Hardware background: [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx), [PCIe x16 capacitor mod](https://170th-street.gitbook.io/hx/modifications/pcie-capacitor-mod), [cmp170hx-linux-p2p](https://github.com/satspace-cpu/cmp170hx-linux-p2p).
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport), the pipeline-parallel fix from [clee/vllm-sm80](https://github.com/clee/vllm-sm80), and [vLLM PR #57105](https://github.com/vllm-project/vllm/pull/57105). The SSD row reader is adapted from clee/vllm-sm80's `ple_preadv.py`.
- Ideas from [Strata](https://github.com/Niko1221/Strata) (bigger prompt chunks, prefetching the table, eager kernel loading, teacher-forced quality checks) and the [DUFRelic recipe](https://github.com/DUFRelic/qwen38-flash-next-recipe).

## License

Apache-2.0 (see `LICENSE`), matching the vLLM and clee/vllm-sm80 code the patch and reader build on. The model and
the GPU driver carry their own licenses.
