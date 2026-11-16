"""Static checks on a parsed GGUF file. Nothing here loads or runs the model."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from .parser import GGUFFile

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Finding:
    id: str
    severity: str  # high | medium | low | info
    title: str
    detail: str
    where: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


# Jinja constructs with no place in a chat template. They are the building blocks of
# server-side template injection; a GGUF chat template that reached an unsandboxed Jinja
# environment gave code execution in llama-cpp-python (CVE-2024-34359).
SSTI_PATTERNS = [
    (r"__\w+__", "double-underscore attribute access"),
    (r"\|\s*attr\s*\(", "attr() filter"),
    (r"\b(popen|subprocess|os\.system|eval|exec|compile|__import__)\b", "process or code execution name"),
    (r"\b(lipsum|cycler|joiner)\b", "Jinja global often used to reach Python builtins"),
    (r"\bgetattr\s*\(", "getattr()"),
    (r"\{%-?\s*(include|import|from|extends)\b", "template include/import"),
]
URL_RE = re.compile(r"https?://[^\s'\"<>)}]+", re.I)
STRING_LITERAL_RE = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", re.S)
# Words model publishers use to label safety-related modification. Self-declared only.
MODIFICATION_WORDS = [
    "abliterated", "abliteration", "uncensored", "unfiltered", "decensored", "jailbreak",
    "jailbroken", "lorablated", "obliterated", "heretic", "no-refusal", "norefusal",
]
NAME_KEYS = [
    "general.name", "general.basename", "general.finetune", "general.description",
    "general.tags", "general.organization", "general.repo_url", "general.url",
]
TEMPLATE_PREFIX = "tokenizer.chat_template"
LONG_LITERAL = 200


def _invisible(text: str) -> list[str]:
    """Zero-width, bidirectional-control and other format characters, as U+XXXX names."""
    found = []
    for ch in text:
        if unicodedata.category(ch) == "Cf" or ch in "  ":
            label = f"U+{ord(ch):04X} {unicodedata.name(ch, 'UNNAMED')}"
            if label not in found:
                found.append(label)
    return found


def check_structure(g: GGUFFile) -> list[Finding]:
    out = [
        Finding("S001", "high", "Structural inconsistency", p, "file layout") for p in g.problems
    ]
    if g.version > 3:
        out.append(Finding("S002", "low", "Newer GGUF version than this tool knows",
                           f"Version {g.version}; checks were written against versions 2 and 3.", "header"))
    if not g.tensors:
        out.append(Finding("S003", "info", "File contains no tensors",
                           "Metadata-only GGUF (for example a vocabulary file).", "header"))
    return out


def check_templates(g: GGUFFile) -> list[Finding]:
    out: list[Finding] = []
    templates = {k: v for k, v in g.metadata.items() if k.startswith(TEMPLATE_PREFIX) and isinstance(v, str)}
    for key, tpl in templates.items():
        for pattern, label in SSTI_PATTERNS:
            hits = sorted({m.group(0) for m in re.finditer(pattern, tpl)})
            if hits:
                out.append(Finding(
                    "T001", "high", "Chat template uses a construct associated with template injection",
                    f"{label}: {', '.join(hits[:6])}. Chat templates only need loops, conditionals and "
                    "string joins. Render this file only with a sandboxed Jinja environment "
                    "(see CVE-2024-34359).", key))
        urls = sorted(set(URL_RE.findall(tpl)))
        if urls:
            out.append(Finding("T002", "medium", "Chat template contains a URL",
                               f"{', '.join(urls[:5])}. A template that places a URL into every prompt can "
                               "steer the model to cite or fetch it.", key))
        inv = _invisible(tpl)
        if inv:
            out.append(Finding("T003", "medium", "Chat template contains invisible characters",
                               f"{'; '.join(inv[:6])}. Text hidden from a human reviewer still reaches the model.", key))
        literals = [a or b for a, b in STRING_LITERAL_RE.findall(tpl)]
        long_ones = [s for s in literals if len(s) >= LONG_LITERAL]
        if long_ones:
            preview = long_ones[0][:160].replace("\n", " ")
            out.append(Finding("T004", "low", "Chat template embeds long fixed text",
                               f"{len(long_ones)} string literal(s) of {LONG_LITERAL}+ characters, e.g. "
                               f"\"{preview}...\". Built-in system prompts are common and usually benign; "
                               "read it, since it is added to every conversation.", key))
    if not templates and g.tensors:
        out.append(Finding("T005", "info", "No chat template in metadata",
                           "The runtime will fall back to its own default prompt format.", TEMPLATE_PREFIX))
    return out


def check_metadata(g: GGUFFile) -> list[Finding]:
    out: list[Finding] = []
    md = g.metadata
    if g.tensors:
        missing = [k for k in ("general.architecture", "general.name") if not md.get(k)]
        if missing:
            out.append(Finding("M001", "low", "Basic identification metadata is missing",
                               f"Absent: {', '.join(missing)}.", "general.*"))
        provenance = [k for k in md if k.startswith(("general.base_model.", "general.source.", "general.license"))
                      or k in ("general.url", "general.repo_url", "general.author", "general.organization")]
        if not provenance:
            out.append(Finding("M002", "info", "No provenance metadata",
                               "No licence, author, source or base-model fields. Provenance must be "
                               "established from where the file was downloaded.", "general.*"))
    declared = []
    for key in NAME_KEYS:
        val = md.get(key)
        text = " ".join(map(str, val)) if isinstance(val, list) else str(val or "")
        for word in MODIFICATION_WORDS:
            if word in text.lower().replace("_", "-"):
                declared.append(f"'{word}' in {key}")
        inv = _invisible(text)
        if inv:
            out.append(Finding("M004", "low", "Invisible characters in a descriptive field",
                               f"{'; '.join(inv[:4])}. Can make two different names look identical.", key))
    if declared:
        out.append(Finding("M003", "info", "Publisher labels this model as safety-modified",
                           f"{'; '.join(sorted(set(declared)))}. This reads the label only: it does not "
                           "test the weights, and an unlabelled file can be modified too.", "general.*"))
    return out


def scan(g: GGUFFile) -> list[Finding]:
    findings = check_structure(g) + check_templates(g) + check_metadata(g)
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.id))


def describe(g: GGUFFile, max_array: int = 5, max_text: int = 300) -> dict[str, Any]:
    """Metadata with long values shortened, plus a tensor-type census."""
    meta: dict[str, Any] = {}
    for k, v in g.metadata.items():
        if isinstance(v, list):
            meta[k] = {"type": "array", "length": len(v), "first": v[:max_array]}
        elif isinstance(v, str) and len(v) > max_text:
            meta[k] = v[:max_text] + f"... ({len(v)} chars)"
        else:
            meta[k] = v
    census: dict[str, int] = {}
    for t in g.tensors:
        census[t.type_name] = census.get(t.type_name, 0) + 1
    return {
        "path": str(g.path),
        "file_size": g.file_size,
        "gguf_version": g.version,
        "alignment": g.alignment,
        "tensor_count": len(g.tensors),
        "tensor_types": census,
        "parameters": sum(t.elements for t in g.tensors),
        "metadata": meta,
    }
