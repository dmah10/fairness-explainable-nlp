from transformers import AutoTokenizer
import torch
import evaluate as hf_eval
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, pipeline
from datasets import load_metric
import numpy as np


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    load_conf = hf_eval.load("confusion_matrix")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="macro")[
        "f1"
    ]
    cm = load_conf.compute(predictions=predictions, references=labels)[
        "confusion_matrix"
    ]
    # numbers true-predicted
    # gives [[0-0, 0-1], [1-0, 1-1]]
    tpr = cm[1][1] / (sum(cm[1]))
    tnr = cm[0][0] / (sum(cm[0]))

    # avreage prediction difference
    #   mean difference between the logits for the two labels
    logits = torch.softmax(torch.tensor(logits), dim=1)
    female_indices = torch.tensor(labels) == 0
    male_indices = torch.tensor(labels) == 1
    female_logits = logits[:, 0][female_indices]
    male_logits = logits[:, 1][male_indices]
    apd = torch.mean(torch.abs(male_logits - female_logits)).item()

    return {"accuracy": accuracy, "f1": f1, "tpr": tpr, "tnr": tnr, "apd": apd}


def preprocess_function(tokenizer):
    return lambda examples: tokenizer(
        examples["text"], truncation=True, max_length=512, padding="longest"
    )


def load_pipeline(model_str, device):
    model = pipeline(
        "text-classification", model=model_str, return_all_scores=True, device=device
    )
    return model


def evaluate(
    model, testset, in_key="text", out_key="label", pred_key="score", max_length=512
):
    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
    preds = model(testset[in_key], **tokenizer_kwargs)
    pred_labels = [
        max(range(len(pred)), key=lambda i: pred[i][pred_key]) for pred in preds
    ]
    score = (np.array(pred_labels) == np.array(testset[out_key])).sum()
    return score / len(testset)
