"""SSD-backed reader for TinyBeast's INT4 PLE n-gram table.

The checkpoint's 128 ``ngram_embedding.shard_N.weight`` tensors already hold the
table exactly as the GPU uses it: each row is 100 bytes, 80 bytes of packed
INT4 (160 values) followed by 10 fp16 group scales (group size 16). Rows are
read positionally from those safetensors files with preadv, so the 29.8 GiB
table never has to live in VRAM or RAM; the page cache keeps the hot rows.

Row r lives in shard ``r // shard_rows`` at local index ``r % shard_rows``,
the same mapping load_weights uses (checkpoint_start = shard * shard_size).
Adapted from clee/vllm-sm80 ple_preadv.py (Apache-2.0), itself a port of
tonyd2wild/Qwen38-Flash-Next-4x3090 ple_mmap.py.
"""
from __future__ import annotations

import json
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor

import torch

PACKED_BYTES = 80
SCALE_BYTES = 20
ROW_BYTES = PACKED_BYTES + SCALE_BYTES


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class PLESSDTable:
    def __init__(self, model_dir: str, layer_idx: int = 1, *, threads: int = 32, chunk_rows: int = 8):
        self.threads = threads
        self.chunk_rows = chunk_rows
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ple-ssd")
        pat = re.compile(rf"(?:^|\.)layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted({v for k, v in weight_map.items() if pat.search(k)})
        shards: dict[int, tuple[str, int, int]] = {}
        for fn in files:
            path = os.path.join(model_dir, fn)
            header, base = _read_header(path)
            for name, meta in header.items():
                m = pat.search(name) if name != "__metadata__" else None
                if not m:
                    continue
                rows, width = meta["shape"]
                if meta["dtype"] != "U8" or width != ROW_BYTES:
                    raise ValueError(f"{name}: expected U8 rows of {ROW_BYTES} bytes, got {meta['dtype']} x {width}")
                begin, end = meta["data_offsets"]
                if end - begin != rows * ROW_BYTES:
                    raise ValueError(f"{name}: size mismatch")
                shards[int(m.group(1))] = (path, base + begin, rows)
        if sorted(shards) != list(range(len(shards))) or not shards:
            raise ValueError("PLE shards missing or not contiguous")
        self.num_shards = len(shards)
        self.shard_rows = shards[0][2]
        if any(shards[i][2] != self.shard_rows for i in range(self.num_shards - 1)):
            raise ValueError("PLE shards are not equal-sized")
        self.rows_total = sum(s[2] for s in shards.values())
        fds: dict[str, int] = {}
        self._base: list[tuple[int, int]] = []
        for i in range(self.num_shards):
            path, begin, _ = shards[i]
            if path not in fds:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                except (AttributeError, OSError):
                    pass
                fds[path] = fd
            self._base.append((fds[path], begin))
        self._staging: torch.Tensor | None = None

    def describe(self) -> str:
        return (f"PLESSDTable(shards={self.num_shards}, rows={self.rows_total:,}, "
                f"row_bytes={ROW_BYTES}, {self.rows_total * ROW_BYTES / 2**30:.1f} GiB, threads={self.threads})")

    def _staging_for(self, n: int, pin: bool) -> torch.Tensor:
        if self._staging is None or self._staging.shape[0] < n:
            self._staging = torch.empty((max(n, 4096), ROW_BYTES), dtype=torch.uint8, pin_memory=pin)
        return self._staging[:n]

    def _gather_chunk(self, ids: list[int], mv: memoryview, start: int) -> None:
        sr, base, preadv = self.shard_rows, self._base, os.preadv
        for k, i in enumerate(ids):
            s, local = divmod(i, sr)
            fd, off = base[s]
            dst = (start + k) * ROW_BYTES
            if preadv(fd, [mv[dst:dst + ROW_BYTES]], off + local * ROW_BYTES) != ROW_BYTES:
                raise IOError(f"short PLE row read (row {i})")

    def gather(self, ids: torch.Tensor, *, pin: bool = True, out: torch.Tensor | None = None) -> torch.Tensor:
        """Flat int64 CPU row ids -> uint8 [n, 100] (packed INT4 | fp16 scales), pinned by default.
        `out` (uint8 [>= n, 100], contiguous) lets the caller rotate its own pinned buffers."""
        flat = ids.reshape(-1)
        n = flat.numel()
        out = self._staging_for(n, pin) if out is None else out[:n]
        if n == 0:
            return out
        if int(flat.min()) < 0 or int(flat.max()) >= self.rows_total:
            raise IndexError("PLE row id out of range")
        id_list = flat.tolist()
        mv = memoryview(out.numpy()).cast("B")
        c = self.chunk_rows
        if n <= c:
            self._gather_chunk(id_list, mv, 0)
        else:
            for f in [self._pool.submit(self._gather_chunk, id_list[a:a + c], mv, a) for a in range(0, n, c)]:
                f.result()
        return out

    @staticmethod
    def split(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[n, 100] uint8 -> (packed [n, 80] uint8, scales [n, 10] fp16), both contiguous."""
        return rows[:, :PACKED_BYTES].contiguous(), rows[:, PACKED_BYTES:].contiguous().view(torch.float16)
