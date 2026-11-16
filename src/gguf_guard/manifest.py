"""Per-tensor integrity manifests, verification and file-to-file diff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import __version__
from .parser import GGUFFile

SKIP_HASH_KEYS = ()  # every metadata key takes part in the metadata digest


def _metadata_digests(g: GGUFFile) -> dict[str, str]:
    return {
        k: hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()
        for k, v in g.metadata.items()
    }


def file_sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build(g: GGUFFile) -> dict[str, Any]:
    """Hash every tensor and every metadata value.

    A whole-file hash only says 'something differs'. This says which tensors and
    which metadata keys differ, so a re-quantised, re-templated or weight-edited
    file can be told apart.
    """
    return {
        "format": "gguf-guard-manifest/1",
        "tool": f"gguf-guard {__version__}",
        "file": g.path.name,
        "file_size": g.file_size,
        "file_sha256": file_sha256(g.path),
        "gguf_version": g.version,
        "metadata": _metadata_digests(g),
        "tensors": {
            t.name: {"type": t.type_name, "dims": t.dims, "bytes": t.size, "sha256": g.tensor_sha256(t)}
            for t in g.tensors
        },
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Differences between two manifests (a = reference, b = candidate)."""
    ta, tb = a["tensors"], b["tensors"]
    ma, mb = a["metadata"], b["metadata"]
    changed, reshaped = [], []
    for name in sorted(set(ta) & set(tb)):
        x, y = ta[name], tb[name]
        if x["type"] != y["type"] or x["dims"] != y["dims"]:
            reshaped.append({"tensor": name, "reference": f"{x['type']} {x['dims']}",
                             "candidate": f"{y['type']} {y['dims']}"})
        elif x["sha256"] != y["sha256"]:
            changed.append(name)
    shared = set(ta) & set(tb)
    result = {
        "identical": a["file_sha256"] == b["file_sha256"],
        "tensors": {
            "total_reference": len(ta),
            "total_candidate": len(tb),
            "unchanged": len(shared) - len(changed) - len(reshaped),
            "changed": changed,
            "type_or_shape_changed": reshaped,
            "only_in_reference": sorted(set(ta) - set(tb)),
            "only_in_candidate": sorted(set(tb) - set(ta)),
        },
        "metadata": {
            "changed": sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k]),
            "only_in_reference": sorted(set(ma) - set(mb)),
            "only_in_candidate": sorted(set(mb) - set(ma)),
        },
    }
    t = result["tensors"]
    notes = []
    if reshaped and not changed and len(reshaped) > len(shared) / 2:
        notes.append("Most tensors changed type or shape: the files are probably different "
                     "quantisations, so byte-level comparison of weights is not meaningful.")
    elif changed and len(changed) <= max(1, len(shared) // 2):
        notes.append(f"{len(changed)} of {len(shared)} shared tensors differ while the rest are "
                     "byte-identical: the candidate is an edit of the reference, not a re-export.")
    if any(k.startswith("tokenizer.chat_template") for k in result["metadata"]["changed"]):
        notes.append("The chat template differs: prompts will be formatted differently.")
    if not (changed or reshaped or t["only_in_reference"] or t["only_in_candidate"]):
        notes.append("All tensors are byte-identical.")
    result["notes"] = notes
    result["limitations"] = [
        "A diff shows that bytes differ, not what the change does to model behaviour.",
        "Only meaningful between files of the same quantisation.",
    ]
    return result


def verify(g: GGUFFile, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("format") != "gguf-guard-manifest/1":
        raise ValueError("not a gguf-guard manifest (format field missing or unknown)")
    diff = compare(manifest, build(g))
    t, m = diff["tensors"], diff["metadata"]
    ok = not any([t["changed"], t["type_or_shape_changed"], t["only_in_reference"],
                  t["only_in_candidate"], m["changed"], m["only_in_reference"], m["only_in_candidate"]])
    return {"verified": ok, **diff}
