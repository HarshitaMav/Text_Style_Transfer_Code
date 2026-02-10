#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import evaluate

# plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained FLAN-T5 style-transfer model")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/flan_t5_base_style_transfer",
        help="Path to the trained model directory (Trainer output_dir).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed_data/test.jsonl",
        help="Path to the test JSONL file with fields 'input' and 'target'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="Max length for input sequences.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Max length for generated sequences.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for generation.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where plots and optional files will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Load model + tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Load test dataset
    logger.info(f"Loading test data from {args.test_file}")
    ds = load_dataset("json", data_files=args.test_file)["train"]

    all_preds = []
    all_labels = []

    logger.info(f"Evaluating on {len(ds)} examples...")
    for start_idx in range(0, len(ds), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(ds))
        batch = ds[start_idx:end_idx]

        inputs = tokenizer(
            batch["input_text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=args.max_target_length,
                num_beams=args.num_beams,
            )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # For labels, we re-tokenize to ensure proper decoding
        label_ids = tokenizer(
            batch["target_text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_target_length,
        )["input_ids"]
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        all_preds.extend([p.strip() for p in preds])
        all_labels.extend([l.strip() for l in labels])

    # Compute metrics
    logger.info("Computing ROUGE and BLEU...")
    rouge_results = rouge_metric.compute(
        predictions=all_preds,
        references=all_labels,
        use_stemmer=True,
    )
    bleu_results = bleu_metric.compute(
        predictions=all_preds,
        references=all_labels,
    )

    metrics = {
        "ROUGE-1": rouge_results["rouge1"],
        "ROUGE-2": rouge_results["rouge2"],
        "ROUGE-L": rouge_results["rougeL"],
        "ROUGE-Lsum": rouge_results.get("rougeLsum", 0.0),
        "BLEU": bleu_results["bleu"],
    }

    print("Evaluation Metrics:\n")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics to a text file
    metrics_path = os.path.join(args.results_dir, "eval_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    logger.info(f"Saved metrics to {metrics_path}")

    # bar plot
    if HAS_MATPLOTLIB:
        logger.info("Generating metrics bar plot...")
        plt.figure()
        names = list(metrics.keys())
        values = [metrics[k] for k in names]
        plt.bar(names, values)
        plt.ylim(0.0, 1.0)
        plt.title("Evaluation Metrics (Test Set)")
        plt.ylabel("Score")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        plot_path = os.path.join(args.results_dir, "eval_metrics.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved metrics plot to {plot_path}")
    else:
        logger.warning("matplotlib not installed; skipping metric plots.")


if __name__ == "__main__":
    main()
