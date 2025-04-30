import os, sys
from itertools import product
import pickle
from pathlib import Path
from copy import copy

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from tqdm import tqdm
import argparse
import wandb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import torch
from ferret import Benchmark
from ferret.evaluators.evaluation import Evaluation
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import (
    TrainingArguments,
    Trainer,
    pipeline,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    T5ForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from sklearn.model_selection import train_test_split

from src.data import (
    load_raw,
    convert_to_hf,
    genderify,
    load_gecobench,
    resplit_gecobench,
)
from src.train import load_pipeline, evaluate, compute_metrics, preprocess_function
from src.util import df_from_ferret_scores, add_special_all_special_tokens
from src.metrics import gini_index, threshold_sparsity, mass_acc, Sensitivity_Evaluation

# Parse arguments
parser = argparse.ArgumentParser(description="Fairness XAI")
parser.add_argument(
    "--data-dir", type=str, default="./data", help="Directory to save the data"
)
parser.add_argument(
    "--project", type=str, default="fairness", help="Wandb project name"
)
parser.add_argument(
    "--split-idx", type=int, default=2, help="Index of the dataset split"
)
parser.add_argument("--train-size", type=int, default=20_000, help="Train set size")
parser.add_argument("--test-size", type=int, default=10_000, help="Test set size")
parser.add_argument(
    "--max-length", type=int, default=1_000_000, help="Max input length in char"
)
parser.add_argument(
    "--min-length", type=int, default=0, help="Min input length in char"
)
parser.add_argument(
    "--epochs", type=int, nargs="*", default=[1], help="Num. train epochs"
)
parser.add_argument("--reps", type=int, default=1, help="Experiment repeats")
parser.add_argument("--batch-size", type=int, default=16, help="Train batch size")
parser.add_argument("--wandb", type=str, help="Wandb key", default=None)
parser.add_argument(
    "--seeds", type=int, nargs="*", help="seeds", default=[1, 2, 3, 4, 5]
)
parser.add_argument(
    "--train",
    action="store_true",
    help="Splits the data into test/train and fine-tunes the model",
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Locally save results at the end",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--categories",
    type=str,
    nargs="*",
    default=["fashion"],
    help="Amazon review categories.",
)
parser.add_argument(
    "--models",
    type=str,
    nargs="*",
    default=["huawei-noah/TinyBERT_General_4L_312D"],
    help="Huggingface model name",
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="*",
    default=["amazon"],
    help="Dataset",
    choices=["amazon", "gecobench", "gecobench-subject"],
)
parser.add_argument(
    "--random",
    action="store_true",
    help="Assign random labels",
)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("running on ", device)
print(args)

for model_str, dataset_str, epochs, r in product(
    args.models, args.datasets, args.epochs, range(args.reps)
):
    seed = args.seeds[r]

    if args.wandb is not None:
        wandb.login(
            relogin=True,
            key=args.wandb,
        )
        wandb.init(
            entity="xai-fairness",
            project=args.project,
            dir=Path(".").resolve(),
            reinit=True,
        )
        config = wandb.config
        config.categories = args.categories
        config.seed = args.seed
        config.model = model_str
        config.max_length = args.max_length
        config.min_length = args.min_length
        config.fine_tuned = args.train
        config.dataset = dataset_str
        if args.train:
            config.epochs = epochs
            config.batch_size = args.batch_size

    min_length, max_length = args.min_length, args.max_length

    if "gecobench" in dataset_str:
        train_path = (
            args.data_dir
            + f'/gecobench/train{"_subject" if "subject" in dataset_str else ""}.jsonl'
        )
        test_path = (
            args.data_dir
            + f'/gecobench/test{"_subject" if "subject" in dataset_str else ""}.jsonl'
        )
        train_df = load_gecobench(Path(train_path), return_df=True)
        test_df = load_gecobench(Path(test_path), return_df=True)

        trainset, testset = resplit_gecobench(train_df, test_df)
        n_labels = 2

    print("len train", len(trainset))
    print("len eval", len(testset))

    if "gpt" in model_str:
        model = GPT2ForSequenceClassification.from_pretrained(
            model_str,
            num_labels=n_labels,
            force_download=False,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_str,
            num_labels=n_labels,
            trust_remote_code=True,
            force_download=True,
            ignore_mismatched_sizes=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_str,
        trust_remote_code=True,
        force_download=False,
    )

    if "gpt" in model_str or "t5" in model_str:
        # tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.mask_token_id = tokenizer.pad_token_id
        tokenizer.mask_token = tokenizer.eos_token
        tokenizer.sep_token = tokenizer.eos_token

        add_special_all_special_tokens(tokenizer)

        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id

    tokenized_train = trainset.map(preprocess_function(tokenizer), batched=True)
    tokenized_test = testset.map(preprocess_function(tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_set = testset

    if args.train:
        training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            save_steps=1e99,
            evaluation_strategy="epoch",
            save_strategy="no",
            #report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        logits, labels, _ = trainer.predict(
            tokenized_test,
        )

        metrics = trainer.compute_metrics((logits, labels))

        if args.wandb is not None:
            wandb.log(metrics)

    if args.wandb is not None:
        name = wandb.run.name
        config.eval_size = len(eval_set)
        config.train_size = len(trainset)
        wandb.finish()
