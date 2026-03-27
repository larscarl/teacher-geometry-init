import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from datasets import load_dataset
from transformers import AutoTokenizer


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue

        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()

        if override or key not in os.environ:
            os.environ[key] = value


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _prepare_split(
    *,
    hf_path: str,
    hf_config: str,
    split: str,
    streaming: bool,
    max_documents: int,
    tokenizer,
    chunk_length: int,
    add_eos_between_docs: bool,
    text_column: str,
    shuffle: bool,
    seed: int,
    streaming_shuffle_buffer: int,
    output_path: Path,
    hf_token: str,
) -> Dict[str, int]:
    if max_documents <= 0:
        raise ValueError("max_documents must be > 0")

    kwargs = {
        "path": hf_path,
        "split": split,
        "streaming": streaming,
    }
    if hf_config:
        kwargs["name"] = hf_config
    if hf_token:
        kwargs["token"] = hf_token

    ds = load_dataset(**kwargs)

    if shuffle:
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=streaming_shuffle_buffer)
        else:
            ds = ds.shuffle(seed=seed)

    eos_id = tokenizer.eos_token_id
    token_buffer = []

    docs_seen = 0
    docs_used = 0
    chunks_written = 0
    tokens_consumed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as wf:
        for row in ds:
            docs_seen += 1
            if docs_used >= max_documents:
                break

            text = row.get(text_column)
            if text is None:
                continue
            text = str(text)
            if not text:
                continue

            ids = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if not ids:
                continue

            if add_eos_between_docs and eos_id is not None:
                ids = ids + [int(eos_id)]

            docs_used += 1
            tokens_consumed += len(ids)
            token_buffer.extend(ids)

            while len(token_buffer) >= chunk_length:
                chunk = token_buffer[:chunk_length]
                del token_buffer[:chunk_length]
                wf.write(json.dumps({"input_ids": chunk}, ensure_ascii=True) + "\n")
                chunks_written += 1

    return {
        "docs_seen": docs_seen,
        "docs_used": docs_used,
        "chunks_written": chunks_written,
        "tokens_consumed": tokens_consumed,
        "tokens_discarded_tail": len(token_buffer),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local tokenized/chunked FineWeb subsets for distillation "
            "(train.jsonl + eval.jsonl with input_ids)."
        )
    )
    parser.add_argument(
        "--hf-path",
        default="HuggingFaceFW/fineweb",
        help="Hugging Face dataset path.",
    )
    parser.add_argument(
        "--hf-config",
        default="sample-10BT",
        help="Hugging Face dataset config/subset (e.g. sample-10BT, CC-MAIN-2024-18).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="HF split for train data.",
    )
    parser.add_argument(
        "--eval-split",
        default="train",
        help="HF split for eval data.",
    )
    parser.add_argument(
        "--train-documents",
        type=int,
        default=20000,
        help="Number of source documents to consume for train.",
    )
    parser.add_argument(
        "--eval-documents",
        type=int,
        default=2000,
        help="Number of source documents to consume for eval.",
    )
    parser.add_argument(
        "--tokenizer-model",
        required=True,
        help="Tokenizer model used for chunking.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=1024,
        help="Target tokens per sequence.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for train.jsonl / eval.jsonl / PREPARE_META.json.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Text column name in HF dataset.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large HF datasets.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle source documents before chunking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )
    parser.add_argument(
        "--streaming-shuffle-buffer",
        type=int,
        default=10000,
        help="Buffer size for streaming shuffle.",
    )
    parser.add_argument(
        "--add-eos-between-docs",
        action="store_true",
        help="Insert tokenizer EOS token between documents.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _load_env_file(Path(".env"), override=False)
    hf_token = os.environ.get("HF_TOKEN", "")

    if args.chunk_length <= 1:
        raise ValueError("--chunk-length must be > 1")
    if args.train_documents <= 0 or args.eval_documents <= 0:
        raise ValueError("--train-documents and --eval-documents must be > 0")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, token=hf_token or None)

    out_dir = Path(args.output_dir)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"

    print(
        "Preparing FineWeb subsets: "
        f"path={args.hf_path} config={args.hf_config} streaming={args.streaming}"
    )
    print(f"Tokenizer={args.tokenizer_model} chunk_length={args.chunk_length}")

    train_stats = _prepare_split(
        hf_path=args.hf_path,
        hf_config=args.hf_config,
        split=args.train_split,
        streaming=bool(args.streaming),
        max_documents=args.train_documents,
        tokenizer=tokenizer,
        chunk_length=args.chunk_length,
        add_eos_between_docs=bool(args.add_eos_between_docs),
        text_column=args.text_column,
        shuffle=bool(args.shuffle),
        seed=args.seed,
        streaming_shuffle_buffer=args.streaming_shuffle_buffer,
        output_path=train_path,
        hf_token=hf_token,
    )

    eval_stats = _prepare_split(
        hf_path=args.hf_path,
        hf_config=args.hf_config,
        split=args.eval_split,
        streaming=bool(args.streaming),
        max_documents=args.eval_documents,
        tokenizer=tokenizer,
        chunk_length=args.chunk_length,
        add_eos_between_docs=bool(args.add_eos_between_docs),
        text_column=args.text_column,
        shuffle=bool(args.shuffle),
        seed=args.seed + 1,
        streaming_shuffle_buffer=args.streaming_shuffle_buffer,
        output_path=eval_path,
        hf_token=hf_token,
    )

    meta = {
        "created_at": _now_iso(),
        "hf_path": args.hf_path,
        "hf_config": args.hf_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "streaming": bool(args.streaming),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "tokenizer_model": args.tokenizer_model,
        "chunk_length": int(args.chunk_length),
        "text_column": args.text_column,
        "add_eos_between_docs": bool(args.add_eos_between_docs),
        "train_documents_target": int(args.train_documents),
        "eval_documents_target": int(args.eval_documents),
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "train_output_path": str(train_path),
        "eval_output_path": str(eval_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PREPARE_META.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    print(
        "Done. "
        f"train_chunks={train_stats['chunks_written']} eval_chunks={eval_stats['chunks_written']}"
    )
    print(f"Wrote: {train_path}")
    print(f"Wrote: {eval_path}")
    print(f"Wrote: {out_dir / 'PREPARE_META.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
