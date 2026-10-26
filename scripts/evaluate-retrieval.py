#!/usr/bin/env python3
"""Measure offline retrieval without writing to the selected user vault."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PRODUCT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRODUCT_ROOT))

from claude_obsidian.retrieval_eval import EvaluationError, IndexNotReadyError, run_fixture, run_quality


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", help="Prebuilt selected vault; evaluation is read-only")
    parser.add_argument("--queries", help="Explicit local retrieval-queries.v1 JSON labels")
    parser.add_argument("--top", type=int, default=10, help="Ranking depth, with a minimum measured depth of 10")
    parser.add_argument("--fixture", action="store_true", help="Build and evaluate the original source fixture only in TemporaryDirectory")
    args = parser.parse_args(argv)
    if args.fixture:
        if args.vault or args.queries or args.top != 10:
            parser.error("--fixture uses its frozen labels and default depth; omit --vault, --queries, and --top")
    elif not args.vault or not args.queries:
        parser.error("provide both --vault and --queries, or use --fixture")
    try:
        report = run_fixture(PRODUCT_ROOT) if args.fixture else run_quality(args.vault, args.queries, PRODUCT_ROOT, args.top)
    except IndexNotReadyError as exc:
        print(f"RETRIEVAL_INDEX_UNAVAILABLE: {exc}", file=sys.stderr)
        return 10
    except EvaluationError as exc:
        print(f"RETRIEVAL_EVALUATION_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
