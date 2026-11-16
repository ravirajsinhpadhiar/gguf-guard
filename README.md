# gguf-guard

Inspect a GGUF model file before you run it. `gguf-guard` reads the header, metadata and tensor table of a `.gguf` file (the format used by llama.cpp, Ollama and LM Studio) and tells you what is in it, what looks wrong, and exactly which tensors differ from a copy you trust.

It never loads or executes the model, and it has no dependencies.

Illustrative output for a file with a hostile chat template:

```console
$ gguf-guard scan model.gguf
model.gguf: GGUF v3, 291 tensors, 34 metadata keys

[HIGH  ] T001 Chat template uses a construct associated with template injection
         double-underscore attribute access: __globals__, __init__. Chat templates only need
         loops, conditionals and string joins. Render this file only with a sandboxed Jinja
         environment (see CVE-2024-34359).
         at: tokenizer.chat_template
```

## Why

A GGUF file is more than weights. It carries a Jinja chat template that the runtime renders on every request, free-text metadata that tools display, and a tensor table that tells the loader where to read. Each is an attack surface:

- **The chat template is code.** An unsandboxed Jinja render of a template from a model file led to remote code execution in llama-cpp-python ([CVE-2024-34359](https://nvd.nist.gov/vuln/detail/CVE-2024-34359)). A template can also quietly add instructions or a URL to every conversation.
- **A file hash tells you little.** Two downloads of "the same model" with different SHA-256 values might differ in one metadata string or in a hundred weight tensors. You need to know which.
- **Names lie.** Invisible Unicode in a model name makes two different files look identical in a list.

## Commands

| Command | What it does |
|---|---|
| `scan FILE` | All static checks. Exit code 1 on a high-severity finding, so it works in CI. |
| `info FILE` | Metadata (long values shortened), tensor count, tensor types, parameter count. |
| `template FILE` | Prints the chat template exactly as stored, for reading. |
| `manifest FILE -o m.json` | SHA-256 of every tensor and every metadata value. |
| `verify FILE m.json` | Checks a file against a saved manifest. Exit code 1 on any difference. |
| `diff REFERENCE CANDIDATE` | Which tensors and which metadata keys differ between two files. |
| `mcp` | Runs as an MCP server (`scan_model`, `model_info`, `diff_models`). |

## Checks

| Id | Severity | Finding |
|---|---|---|
| S001 | high | Structural inconsistency: duplicate tensor or metadata names, misaligned or out-of-file tensor offsets, tensors sharing an offset, bad alignment value |
| S002 | low | GGUF version newer than 3 |
| S003 | info | No tensors (metadata-only file) |
| T001 | high | Chat template uses Jinja constructs associated with template injection (`__class__`, `__globals__`, `attr()`, `popen`, `lipsum`, `include`/`import`, and similar) |
| T002 | medium | Chat template contains a URL |
| T003 | medium | Chat template contains invisible characters (zero-width, bidirectional controls) |
| T004 | low | Chat template embeds fixed text of 200+ characters |
| T005 | info | No chat template |
| M001 | low | Missing `general.architecture` or `general.name` |
| M002 | info | No licence, author, source or base-model metadata |
| M003 | info | Publisher labels the model as safety-modified ("abliterated", "uncensored", and similar) |
| M004 | low | Invisible characters in a name or description field |

Malformed files (bad magic, truncated header, absurd counts) are rejected outright with exit code 2. The parser enforces size limits so a hostile header cannot make it allocate gigabytes.

## Install

```bash
pip install git+https://github.com/ravirajsinhpadhiar/gguf-guard
```

Python 3.10 or newer.

## Typical uses

**Before running a model you downloaded:**

```bash
gguf-guard scan ~/models/some-model.Q4_K_M.gguf
gguf-guard template ~/models/some-model.Q4_K_M.gguf     # read what gets added to every prompt
```

**Pin a model you have vetted, and detect any later change:**

```bash
gguf-guard manifest model.gguf -o model.manifest.json
gguf-guard verify model.gguf model.manifest.json        # in CI or before each deployment
```

**Find out how two builds differ:**

```bash
gguf-guard diff original.gguf suspicious.gguf
```

Illustrative output (shortened):

```json
{
  "tensors": { "unchanged": 289, "changed": ["blk.11.attn_output.weight", "blk.12.attn_output.weight"] },
  "metadata": { "changed": ["tokenizer.chat_template"] },
  "notes": ["2 of 291 shared tensors differ while the rest are byte-identical: the candidate is an edit of the reference, not a re-export."]
}
```

Ollama stores models as GGUF blobs under `~/.ollama/models/blobs/`; point `gguf-guard` at the largest `sha256-…` file for a model.

## What it does not do

- **It does not judge the weights.** A clean scan does not mean a model is unmodified, unbiased or free of backdoors. `diff` shows *that* tensors differ from a reference, not what the change does to behaviour. Check M003 reads the publisher's own label only.
- `diff` and `verify` are meaningful only between files of the same quantisation; a different quantisation changes every tensor.
- Tensor hashes cover each tensor's bytes up to the next tensor, including alignment padding.
- Big-endian GGUF and GGUF version 1 are not supported.

## Develop

```bash
git clone https://github.com/ravirajsinhpadhiar/gguf-guard && cd gguf-guard
pip install -e ".[dev]"
pytest
```

Test models are generated with llama.cpp's reference `gguf` package, so the parser is checked against the reference writer. The parser was also compared with the reference reader on the vocabulary files shipped in the llama.cpp repository.

## Licence

MIT.
