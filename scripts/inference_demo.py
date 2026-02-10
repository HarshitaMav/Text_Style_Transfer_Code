#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


DEFAULT_STYLES = ["professional", "casual", "polite", "social", "emojify"]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference demo for multi-style text generation")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/flan_t5_base_style_transfer",
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input sentence to rewrite.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=DEFAULT_STYLES,
        help="List of styles to generate. Default: professional casual polite social emojify",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Rewrite the following sentence in a {style} style:\n{text}",
        help="Prompt template. Must contain '{style}' and '{text}'.",
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

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading model and tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    input_text = args.text.strip()
    print("========================================")
    print("Original text:")
    print(input_text)
    print("========================================")

    for style in args.styles:
        prompt = args.prompt_template.format(style=style, text=input_text)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
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

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        print(f"\n[{style.upper()}]")
        print(output)

if __name__ == "__main__":
    main()
