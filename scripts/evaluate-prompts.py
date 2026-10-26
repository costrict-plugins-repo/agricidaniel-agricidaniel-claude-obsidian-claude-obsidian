#!/usr/bin/env python3
"""Build or score the frozen offline Query and Save prompt acceptance packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PRODUCT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRODUCT_ROOT))

from claude_obsidian.prompt_eval import (
    PromptEvaluationError,
    evaluate_response,
    generate_packet,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "packet", help="Print a packet bound to the current Query and Save sources"
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one supplied response without provider execution"
    )
    evaluate_parser.add_argument("response", help="Strict prompt-response.v1 JSON file")
    evaluate_parser.add_argument(
        "--packet",
        help="Previously generated prompt-packet.v1 JSON; current sources are rechecked",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "packet":
            result = generate_packet(PRODUCT_ROOT)
        else:
            packet = args.packet or generate_packet(PRODUCT_ROOT)
            result = evaluate_response(
                packet, args.response, product_root=PRODUCT_ROOT
            )
    except PromptEvaluationError as exc:
        print(f"PROMPT_EVALUATION_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "evaluate" and not result["summary"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
