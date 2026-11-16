from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, checks, manifest
from .parser import GGUFError, parse

MARK = {"high": "HIGH  ", "medium": "MEDIUM", "low": "LOW   ", "info": "INFO  "}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gguf-guard",
        description="Inspect GGUF model files before you run them: structure, chat template, "
        "metadata, and per-tensor integrity.",
    )
    p.add_argument("--version", action="version", version=f"gguf-guard {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="run all static checks")
    s.add_argument("file")
    s.add_argument("--json", action="store_true")
    s.add_argument("--fail-on", choices=["high", "medium", "low", "never"], default="high",
                   help="exit 1 when a finding of this severity or worse exists (default: high)")

    s = sub.add_parser("info", help="metadata and tensor summary")
    s.add_argument("file")

    s = sub.add_parser("template", help="print the chat template(s) exactly as stored")
    s.add_argument("file")

    s = sub.add_parser("manifest", help="write per-tensor SHA-256 hashes")
    s.add_argument("file")
    s.add_argument("-o", "--output", help="manifest path (default: stdout)")

    s = sub.add_parser("verify", help="check a file against a saved manifest")
    s.add_argument("file")
    s.add_argument("manifest")

    s = sub.add_parser("diff", help="tensor-level and metadata-level diff of two files")
    s.add_argument("reference")
    s.add_argument("candidate")

    sub.add_parser("mcp", help="run as an MCP server over stdio")
    return p


def _print(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mcp":
        from .mcp_server import run

        run()
        return 0
    try:
        if args.command == "scan":
            g = parse(args.file)
            findings = checks.scan(g)
            if args.json:
                _print({"file": str(g.path), "gguf_version": g.version, "tensors": len(g.tensors),
                        "findings": [f.as_dict() for f in findings]})
            else:
                print(f"{g.path.name}: GGUF v{g.version}, {len(g.tensors)} tensors, "
                      f"{len(g.metadata)} metadata keys")
                if not findings:
                    print("No findings.")
                for f in findings:
                    print(f"\n[{MARK[f.severity]}] {f.id} {f.title}")
                    print(f"         {f.detail}")
                    if f.where:
                        print(f"         at: {f.where}")
                print("\nStatic checks only: a clean scan does not mean the weights are unmodified or safe.")
            if args.fail_on == "never":
                return 0
            limit = checks.SEVERITY_ORDER[args.fail_on]
            return 1 if any(checks.SEVERITY_ORDER[f.severity] <= limit for f in findings) else 0

        if args.command == "info":
            _print(checks.describe(parse(args.file)))
        elif args.command == "template":
            g = parse(args.file)
            found = {k: v for k, v in g.metadata.items()
                     if k.startswith(checks.TEMPLATE_PREFIX) and isinstance(v, str)}
            if not found:
                print("No chat template in this file.", file=sys.stderr)
                return 1
            for key, tpl in found.items():
                print(f"--- {key} ({len(tpl)} chars)")
                print(tpl)
        elif args.command == "manifest":
            m = manifest.build(parse(args.file))
            if args.output:
                Path(args.output).write_text(json.dumps(m, indent=2), encoding="utf-8")
                print(f"{len(m['tensors'])} tensors hashed -> {args.output}", file=sys.stderr)
            else:
                _print(m)
        elif args.command == "verify":
            result = manifest.verify(parse(args.file), json.loads(Path(args.manifest).read_text("utf-8")))
            _print(result)
            return 0 if result["verified"] else 1
        elif args.command == "diff":
            _print(manifest.compare(manifest.build(parse(args.reference)),
                                    manifest.build(parse(args.candidate))))
    except (GGUFError, ValueError, OSError) as exc:
        print(f"gguf-guard: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
