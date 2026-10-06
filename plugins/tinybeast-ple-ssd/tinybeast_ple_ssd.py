"""Serve TinyBeast's INT4 PLE n-gram table from the SSD instead of VRAM.

Off unless TB_PLE_SSD=1 (and the plugin is listed in VLLM_PLUGINS), so the
production launcher is unaffected by this package being installed.

How it works:
  * The 29.8 GiB INT4 table is never allocated on the GPU: the weight becomes a
    1-row placeholder and the shard loads become no-ops (the checkpoint shards
    on disk ARE the table, see ple_ssd_reader.py).
  * Each step, before the forward runs, Qwen4ExpModelState.prepare_inputs
    computes the step's n-gram row ids, reads those rows from the SSD into
    pinned memory, and copies them into small fixed GPU buffers.
  * The INT4 embedding method then decodes from those buffers with the same
    kernel the GPU-resident table uses (ids = row positions in the buffer), so
    every downstream op (global FP8 scale, ETP select, flatten) is unchanged.
  * prepare_dummy_inputs (profiling, CUDA graph capture) only zeroes the
    buffers, so no SSD reads happen inside capture.

Also (separate switch, TB_TORCH_MEM_CAP_GIB): caps PyTorch's allocator on each
worker's GPU. GPU1 (0c:00) faults with MMU REGION_VIOLATION when usage climbs
into the top of its unlocked 64 GB (2026-10-05: prompts over ~200K tokens crashed
it while the QSA indexer's per-chunk logits buffers grew the allocator cache). At
the cap PyTorch frees its cache and retries instead of taking those pages. It
lives here only because this plugin already runs inside every worker.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

import torch

logger = logging.getLogger("tinybeast_ple_ssd")
_TIMING = os.environ.get("TB_PLE_SSD_TIMING") == "1"
_PREFETCH = os.environ.get("TB_PLE_SSD_PREFETCH") == "1"
_VERIFY = int(os.environ.get("TB_PLE_SSD_VERIFY", "0") or 0)  # verify the first N CPU-id steps


class _CpuNgramIds:
    """Runs a Qwen4ExpNGramEmbedding's own CPU id-hashing path (compute_ngram_ids with CPU inputs) against CPU
    copies of its three small id tables, so prefill steps get their row ids without touching the GPU."""

    def __init__(self, m):
        cls = type(m)
        self.layer_multipliers = m.layer_multipliers.detach().cpu()
        self.ngram_heads_vocab_sizes = m.ngram_heads_vocab_sizes.detach().cpu()
        self.ngram_heads_offsets = m.ngram_heads_offsets.detach().cpu()
        self.eos_token_id, self.ngram_size = m.eos_token_id, m.ngram_size
        self.heads_per_ngram = m.heads_per_ngram
        self._shift_precompute, self._shift_apply = cls._shift_precompute, cls._shift_apply
        self._compute = cls.compute_ngram_ids

    def __call__(self, input_ids, qsl, ctx):
        return self._compute(self, input_ids, qsl, ctx)

PACKED = 80
SCALE_BYTES = 20


def _enabled() -> bool:
    return os.environ.get("TB_PLE_SSD", "0") == "1"


def _apply_torch_mem_cap(device) -> None:
    cap = os.environ.get("TB_TORCH_MEM_CAP_GIB")
    if not cap or not torch.cuda.is_available():
        return
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_memory
    frac = min(1.0, float(cap) * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    logger.warning("TB torch memory cap: %.2f of %.2f GiB on cuda:%d (fraction %.4f)",
                   float(cap), total / 2**30, idx, frac)


def register() -> None:
    if not _enabled():
        return
    import tinybeast_ple  # the INT4 plugin; provides install() and lookup()
    from vllm.model_executor.parameter import ModelWeightParameter
    from vllm.models.qwen4_exp.nvidia import model_state as ms
    from vllm.models.qwen4_exp.nvidia import ngram_embedding as ng

    tinybeast_ple.install()  # idempotent; SSD mode wraps the INT4 method it selects
    if _PREFETCH:
        from vllm.v1.worker.gpu import states as _st
        if not getattr(_st.RequestState, "_tb_tokens_patched", False):
            _orig_add = _st.RequestState.add_request

            def add_request(self, req_id, *args, **kwargs):
                _orig_add(self, req_id, *args, **kwargs)
                toks = kwargs["all_token_ids"] if "all_token_ids" in kwargs else args[1]
                self.__dict__.setdefault("_tb_tokens", {})[self.req_id_to_index[req_id]] = np.asarray(
                    toks, dtype=np.int64)

            _st.RequestState.add_request = add_request
            _st.RequestState._tb_tokens_patched = True
    if getattr(ng, "_tinybeast_ple_ssd_installed", False):
        return

    prev_select = ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def _ssd_method_for(int4_method):
        base = type(int4_method)

        class Int4SSDMethod(base):
            def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                               input_size, output_size, params_dtype, **attrs):
                from vllm.config import get_current_vllm_config

                width = input_size_per_partition
                if width != 160:
                    raise ValueError(f"Unexpected PLE embedding width: {width}")
                vcfg = get_current_vllm_config()
                tcfg = vcfg.model_config.hf_text_config
                heads = (int(tcfg.ngram_size) - 1) * int(tcfg.heads_per_ngram)
                max_rows = (int(vcfg.scheduler_config.max_num_batched_tokens) + 64) * heads
                data = layer.allocate_embedding_weight(1, width // 2, torch.uint8)  # placeholder
                device = data.device
                layer.register_buffer("int4_scales", torch.zeros(1, width // 16, device=device, dtype=torch.float16))
                # Per-step staging, fixed shape/address so captured graphs stay valid.
                layer._ssd_rows = torch.zeros(max_rows, PACKED + SCALE_BYTES, dtype=torch.uint8, device=device)
                layer._ssd_packed = torch.zeros(max_rows, PACKED, dtype=torch.uint8, device=device)
                layer._ssd_scales = torch.zeros(max_rows, width // 16, dtype=torch.float16, device=device)
                layer._ssd_arange = torch.arange(max_rows, dtype=torch.int64, device=device)
                layer._ssd_max_rows = max_rows

                def loader(param, loaded_weight, checkpoint_start=None):
                    return 0  # rows are served from the SSD; shapes are validated by load_weights

                layer.register_parameter("weight", ModelWeightParameter(
                    data=data, input_dim=1, output_dim=0, weight_loader=loader))
                global_scale = ng.create_fp8_scale_parameter(
                    ng.PerTensorScaleParameter, output_partition_sizes, input_size_per_partition,
                    None, attrs.get("weight_loader"), scale_dtype=torch.float32)
                layer.register_parameter("weight_scale", global_scale)
                logger.warning("PLE SSD mode: table served from disk; GPU staging %d rows (%.1f MiB)",
                               max_rows, max_rows * (PACKED + SCALE_BYTES + PACKED + 20 + 8) / 2**20)

            def embedding(self, layer, input_):
                n = input_.numel()
                if n > layer._ssd_max_rows:
                    raise RuntimeError(f"PLE SSD staging holds {layer._ssd_max_rows} rows, step needs {n}")
                ids = layer._ssd_arange[:n].view(input_.shape)
                return tinybeast_ple.lookup(layer._ssd_packed, layer._ssd_scales, ids)

        return Int4SSDMethod()

    def select(quant_config, prefix, embedding_dtype=None):
        return _ssd_method_for(prev_select(quant_config, prefix, embedding_dtype))

    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(select)

    orig_init = ms.Qwen4ExpModelState.__init__
    orig_prepare = ms.Qwen4ExpModelState.prepare_inputs
    orig_dummy = ms.Qwen4ExpModelState.prepare_dummy_inputs

    def __init__(self, vllm_config, model, encoder_cache, device):
        orig_init(self, vllm_config, model, encoder_cache, device)
        _apply_torch_mem_cap(device)
        self._ssd_modules = []
        self._ssd_table = None
        if not getattr(self, "uses_ngram_embedding", False):
            return
        from ple_ssd_reader import PLESSDTable

        mods = [m for m in model.modules()
                if isinstance(m, ng.Qwen4ExpNGramEmbedding) and hasattr(m.ngram_embedding, "_ssd_packed")]
        if not mods:
            return
        ple_ids = sorted(int(i) - 1 for i in self.model_config.hf_text_config.ple_layer_ids)
        if len(ple_ids) != 1 or len(mods) != 1:
            raise RuntimeError(f"PLE SSD mode supports exactly one PLE layer, found {ple_ids} / {len(mods)}")
        table = PLESSDTable(vllm_config.model_config.model, layer_idx=ple_ids[0],
                            threads=int(os.environ.get("TB_PLE_SSD_THREADS", "32")))
        org = int(mods[0].ngram_embedding.org_vocab_size)
        if table.rows_total != org:
            raise RuntimeError(f"PLE SSD table has {table.rows_total} rows, model expects {org}")
        self._ssd_modules, self._ssd_table = mods, table
        logger.warning("PLE SSD mode active: %s", table.describe())

    def _stage(self, m, input_ids, qsl, ctx, n, padded, ids_cpu=None):
        emb = m.ngram_embedding
        h = m.ngram_heads
        k, kp = n * h, padded * h
        if n > 0:
            t0 = time.perf_counter()
            cpu_path = ids_cpu is not None
            if not cpu_path:
                ids = m.compute_ngram_ids(input_ids, qsl, ctx)
                ids_cpu = ids.reshape(-1).to("cpu", dtype=torch.int64)  # syncs the stream
            t1 = time.perf_counter()
            buf = None
            if _PREFETCH:  # no stream sync on the CPU-id path, so rotate two pinned buffers
                bufs = self.__dict__.get("_tb_bufs")
                if bufs is None:
                    rows_cap = emb._ssd_rows.shape[0]
                    bufs = self._tb_bufs = [[torch.empty((rows_cap, PACKED + SCALE_BYTES), dtype=torch.uint8,
                                                         pin_memory=True), None] for _ in range(2)]
                    self._tb_buf_i = 0
                buf = bufs[self._tb_buf_i]
                self._tb_buf_i ^= 1
                if buf[1] is not None:
                    buf[1].synchronize()  # the H2D copy that last read this pinned buffer is done
                rows = self._ssd_table.gather(ids_cpu, out=buf[0])
            else:
                rows = self._ssd_table.gather(ids_cpu)
            if _TIMING and n >= 64:  # prefill-sized steps only
                st = self.__dict__.setdefault("_tb_timing", [0, 0.0, 0.0, 0, 0])
                st[0] += 1; st[1] += t1 - t0; st[2] += time.perf_counter() - t1; st[3] += n; st[4] += cpu_path
                if st[0] % 50 == 0:
                    logger.warning("PLE SSD timing: %d prefill steps (%d via CPU ids), avg %.0f tokens, "
                                   "ids/wait %.1f ms, SSD gather %.1f ms per step", st[0], st[4], st[3] / st[0],
                                   1e3 * st[1] / st[0], 1e3 * st[2] / st[0])
            emb._ssd_rows[:k].copy_(rows, non_blocking=True)
            if buf is not None:
                ev = torch.cuda.Event()
                ev.record()
                buf[1] = ev
            emb._ssd_packed[:k].copy_(emb._ssd_rows[:k, :PACKED])
            emb._ssd_scales.view(torch.uint8)[:k].copy_(emb._ssd_rows[:k, PACKED:])
        if kp > k:
            emb._ssd_packed[k:kp].zero_()
            emb._ssd_scales[k:kp].zero_()

    def _cpu_ids(self, m, input_batch, req_states, n):
        """Row ids for a prefill-only step from CPU data, or None when the GPU path must be used."""
        nr = int(input_batch.num_reqs)
        if not _PREFETCH or self.__dict__.get("_tb_cpu_off") or nr == 0 or n < 64:
            return None
        if int(getattr(input_batch, "num_draft_tokens", 0) or 0) != 0:
            return None
        if not bool(np.all(input_batch.is_prefilling_np[:nr])):
            return None
        toks = getattr(req_states, "_tb_tokens", None)
        if not toks:
            return None
        qsl_np = np.asarray(input_batch.query_start_loc_np[: nr + 1], dtype=np.int64)
        lctx, eos = m.ngram_size - 1, m.eos_token_id
        ctx = np.full((nr, lctx), eos, dtype=np.int64)
        parts = []
        for i in range(nr):
            arr = toks.get(int(input_batch.idx_mapping_np[i]))
            start, cnt = int(input_batch.num_computed_tokens_np[i]), int(qsl_np[i + 1] - qsl_np[i])
            if arr is None or start + cnt > arr.shape[0]:
                return None
            parts.append(arr[start:start + cnt])
            lo = max(0, start - lctx)
            if start > lo:
                ctx[i, lctx - (start - lo):] = arr[lo:start]
        flat = np.concatenate(parts)
        if flat.shape[0] != n:
            return None
        cache = self.__dict__.setdefault("_tb_cpu_ngram", {})
        fn = cache.get(id(m))
        if fn is None:
            fn = cache[id(m)] = _CpuNgramIds(m)
        ids = fn(torch.from_numpy(flat), torch.from_numpy(qsl_np - qsl_np[0]), torch.from_numpy(ctx))
        return ids.reshape(-1).to(torch.int64)

    def prepare_inputs(self, input_batch, req_states):
        model_inputs = orig_prepare(self, input_batch, req_states)
        if self._ssd_modules:
            n = int(input_batch.num_tokens)
            padded = int(input_batch.num_tokens_after_padding)
            num_reqs = int(input_batch.num_reqs)
            qsl = model_inputs["query_start_loc"][: num_reqs + 1]
            ctx = model_inputs["ngram_context"][:num_reqs]
            for m in self._ssd_modules:
                ids_cpu = _cpu_ids(self, m, input_batch, req_states, n)
                if ids_cpu is not None and self.__dict__.get("_tb_verify", [0])[0] < _VERIFY:
                    ref = m.compute_ngram_ids(input_batch.input_ids[:n], qsl, ctx).reshape(-1).to(
                        "cpu", dtype=torch.int64)
                    vs = self.__dict__.setdefault("_tb_verify", [0])
                    vs[0] += 1
                    if not torch.equal(ref, ids_cpu):
                        self._tb_cpu_off = True
                        logger.error("PLE SSD prefetch: CPU ids differ from GPU ids (n=%d, %d of %d differ); "
                                     "CPU path disabled", n, int((ref != ids_cpu).sum()), ref.numel())
                        ids_cpu = ref
                    elif vs[0] in (1, 10, 100) or vs[0] % 250 == 0:
                        logger.warning("PLE SSD prefetch verify: %d prefill steps, CPU ids == GPU ids", vs[0])
                _stage(self, m, input_batch.input_ids[:n], qsl, ctx, n, padded, ids_cpu)
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        model_inputs = orig_dummy(self, num_reqs, num_tokens)
        for m in getattr(self, "_ssd_modules", []):
            emb, k = m.ngram_embedding, int(num_tokens) * m.ngram_heads
            emb._ssd_packed[:k].zero_()
            emb._ssd_scales[:k].zero_()
        return model_inputs

    ms.Qwen4ExpModelState.__init__ = __init__
    ms.Qwen4ExpModelState.prepare_inputs = prepare_inputs
    ms.Qwen4ExpModelState.prepare_dummy_inputs = prepare_dummy_inputs
    ng._tinybeast_ple_ssd_installed = True
    logger.warning("tinybeast_ple_ssd registered (TB_PLE_SSD=1)")
