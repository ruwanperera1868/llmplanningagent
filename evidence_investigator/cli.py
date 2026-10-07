"""Command-line entry point: ``python -m evidence_investigator "question"``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .investigator import Investigator
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evidence_investigator",
                                description="Plan, retrieve, appraise and report on academic evidence.")
    p.add_argument("question", help="the research question, in quotes")
    p.add_argument("--offline", action="store_true",
                   help="use the bundled corpus and no LLM (deterministic, no network)")
    p.add_argument("--max-cycles", type=int, help="maximum plan-search-critique cycles")
    p.add_argument("--out", type=Path, help="write the Markdown briefing to this file")
    p.add_argument("--db", type=Path, help="SQLite database path")
    p.add_argument("-v", "--verbose", action="store_true", help="log each agent step")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {"offline": True} if args.offline else {}
    if args.max_cycles:
        overrides["max_cycles"] = args.max_cycles
    if args.db:
        overrides["db_path"] = args.db
    settings = Settings.from_env(**overrides)

    storage = Storage(settings.db_path)
    try:
        result = Investigator(settings, storage=storage).run(args.question)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        storage.close()

    if args.verbose:
        for e in result.audit:
            print(f"[cycle {e.cycle}] {e.agent:<10} {e.action:<18} {e.detail}", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.markdown, encoding="utf-8")
        print(f"Briefing written to {args.out} (run {result.run_id})")
    else:
        print(result.markdown)
    return 0
