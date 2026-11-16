"""Dependency-free GGUF reader.

Parses the header, metadata and tensor table only; tensor data is never loaded
into memory. Layout follows the GGUF specification in the ggml repository
(https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# Sanity limits. A hostile file can claim absurd counts; refuse instead of allocating.
MAX_KV = 1_000_000
MAX_TENSORS = 1_000_000
MAX_STRING = 256 * 1024 * 1024
MAX_ARRAY = 100_000_000
MAX_DIMS = 8

_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
TYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32", 6: "float32",
    7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}
# ggml tensor types, for display only. Unknown ids are shown as numbers.
TENSOR_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}


class GGUFError(Exception):
    """The file is not valid GGUF, or is truncated or inconsistent."""


@dataclass
class Tensor:
    name: str
    dims: list[int]
    type_id: int
    offset: int  # relative to the start of the tensor data section
    size: int = 0  # bytes up to the next tensor (or end of file); includes alignment padding

    @property
    def type_name(self) -> str:
        return TENSOR_TYPES.get(self.type_id, f"type_{self.type_id}")

    @property
    def elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GGUFFile:
    path: Path
    version: int
    file_size: int
    metadata: dict[str, Any]
    metadata_types: dict[str, str]
    tensors: list[Tensor]
    alignment: int
    data_offset: int
    problems: list[str] = field(default_factory=list)  # structural issues that did not stop parsing

    def tensor_sha256(self, tensor: Tensor, chunk: int = 4 * 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as fh:
            fh.seek(self.data_offset + tensor.offset)
            left = tensor.size
            while left > 0:
                block = fh.read(min(chunk, left))
                if not block:
                    break
                h.update(block)
                left -= len(block)
        return h.hexdigest()


class _Reader:
    def __init__(self, fh: BinaryIO, size: int):
        self.fh, self.size = fh, size

    def take(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError(f"file truncated at offset {self.fh.tell()} (wanted {n} bytes)")
        return data

    def unpack(self, fmt: str, n: int) -> Any:
        return struct.unpack(fmt, self.take(n))[0]

    def u32(self) -> int:
        return self.unpack("<I", 4)

    def u64(self) -> int:
        return self.unpack("<Q", 8)

    def string(self) -> str:
        length = self.u64()
        if length > MAX_STRING or length > self.size:
            raise GGUFError(f"string length {length} at offset {self.fh.tell()} is not plausible")
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, type_id: int, depth: int = 0) -> Any:
        if type_id in _SCALARS:
            fmt, n = _SCALARS[type_id]
            return self.unpack(fmt, n)
        if type_id == 8:
            return self.string()
        if type_id == 9:
            if depth > 4:
                raise GGUFError("arrays nested too deeply")
            inner = self.u32()
            count = self.u64()
            if count > MAX_ARRAY or count > self.size:
                raise GGUFError(f"array length {count} is not plausible")
            if inner in _SCALARS:  # bulk read: far faster for large token-score arrays
                fmt, n = _SCALARS[inner]
                return list(struct.unpack(f"<{count}{fmt[1]}", self.take(n * count)))
            return [self.value(inner, depth + 1) for _ in range(count)]
        raise GGUFError(f"unknown metadata value type {type_id} at offset {self.fh.tell()}")


def parse(path: str | Path) -> GGUFFile:
    p = Path(path).expanduser()
    if not p.is_file():
        raise GGUFError(f"no such file: {path}")
    size = p.stat().st_size
    with open(p, "rb") as fh:
        r = _Reader(fh, size)
        if size < 24:
            raise GGUFError("file is too small to be GGUF")
        magic = r.take(4)
        if magic != MAGIC:
            raise GGUFError(f"not a GGUF file (magic {magic!r}, expected b'GGUF')")
        version = r.u32()
        if version & 0xFFFF == 0 and version != 0:
            raise GGUFError("big-endian GGUF files are not supported")
        if version < 2:
            raise GGUFError(f"GGUF version {version} is not supported (needs 2 or newer)")
        n_tensors, n_kv = r.u64(), r.u64()
        if n_kv > MAX_KV or n_tensors > MAX_TENSORS:
            raise GGUFError(f"implausible header counts (tensors={n_tensors}, metadata={n_kv})")

        problems: list[str] = []
        metadata: dict[str, Any] = {}
        types: dict[str, str] = {}
        for _ in range(n_kv):
            key = r.string()
            type_id = r.u32()
            val = r.value(type_id)
            if key in metadata:
                problems.append(f"duplicate metadata key '{key}' (readers disagree on which value wins)")
            metadata[key] = val
            types[key] = TYPE_NAMES.get(type_id, str(type_id))

        tensors: list[Tensor] = []
        seen: set[str] = set()
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            if n_dims > MAX_DIMS:
                raise GGUFError(f"tensor '{name}' claims {n_dims} dimensions")
            dims = [r.u64() for _ in range(n_dims)]
            type_id, offset = r.u32(), r.u64()
            if name in seen:
                problems.append(f"duplicate tensor name '{name}'")
            seen.add(name)
            tensors.append(Tensor(name, dims, type_id, offset))

        alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            problems.append(f"general.alignment is {alignment!r}; expected a power of two, using 32")
            alignment = DEFAULT_ALIGNMENT
        pos = fh.tell()
        data_offset = pos + (alignment - pos % alignment) % alignment

    data_size = size - data_offset
    if tensors and data_size < 0:
        problems.append("file ends before the tensor data section begins")
    ordered = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(ordered):
        end = ordered[i + 1].offset if i + 1 < len(ordered) else max(data_size, 0)
        t.size = max(end - t.offset, 0)
        if t.offset % alignment:
            problems.append(f"tensor '{t.name}' offset {t.offset} is not aligned to {alignment}")
        if t.offset >= max(data_size, 0):
            problems.append(f"tensor '{t.name}' starts beyond the end of the file")
        elif t.size == 0 and t.elements:
            problems.append(f"tensor '{t.name}' shares its offset with another tensor")

    return GGUFFile(p, version, size, metadata, types, tensors, alignment, data_offset, problems)
