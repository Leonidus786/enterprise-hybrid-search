"""CLI for converting a BEIR dataset into the project's evaluation format."""

from __future__ import annotations

import argparse
import json

from hs.beir import convert_beir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a BEIR dataset split for hybrid-search evaluation."
    )
    parser.add_argument(
        "--beir-dir",
        required=True,
        help="Directory containing corpus.jsonl, queries.jsonl and qrels/",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where converted files will be written",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Qrels split filename without .tsv (default: test)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_parser().parse_args(argv)

    summary = convert_beir(
        beir_dir=args.beir_dir,
        out_dir=args.out_dir,
        split=args.split,
    )

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()