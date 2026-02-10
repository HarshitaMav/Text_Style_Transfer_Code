#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import argparse
from typing import Dict, Any

from datasets import load_dataset

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class TextStyleDataset(Dataset):
    """
    JSONL dataset with fields:
      - "input_text":  str (instruction + source text)
      - "target_text": str (target style text)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        logger.info(f"Loading dataset from {data_path}")
        self.ds = load_dataset("json", data_files=data_path)["train"]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.ds[idx]
        input_text = example["input_text"]
        target_text = example["target_text"]

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,  # dynamic padding via DataCollatorForSeq2Seq
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# -------------------------------------------------------------------------
# Argparse
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a FLAN-T5 model for multi-style text transfer (train-only, no in-loop eval)"
    )

    # Data
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed_data/train.jsonl",
        help="Path to the training JSONL file.",
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/flan-t5-base",
        help="Pretrained model name or path.",
    )

    # Training configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/flan_t5_base_style_transfer",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="Maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Maximum total target sequence length after tokenization.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of update steps to accumulate before a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay if applied.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="If > 0, override num_train_epochs and train for this many total steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X update steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Mixed precision
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed
    set_seed(args.seed)

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer & model
    logger.info(f"Loading tokenizer and model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # Ensure pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # (Optional but safe) disable cache for training to reduce memory spikes
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Load training dataset
    train_dataset = TextStyleDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    # Data collator (handles dynamic padding + labels = -100 for pad)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    # Training arguments: NOTE do_eval=False, no eval/metrics inside training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,  # fine on 4.x
    )

    logger.info("***** Starting training *****")
    train_result = trainer.train()
    trainer.save_model()  # Saves model + tokenizer in output_dir

    metrics = train_result.metrics
    if "train_loss" in metrics:
        logger.info(f"Final training loss: {metrics['train_loss']:.4f}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("***** Training complete; model saved *****")


if __name__ == "__main__":
    main()
