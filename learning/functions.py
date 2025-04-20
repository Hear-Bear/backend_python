from typing import Tuple, List, Dict, Any

import evaluate
import numpy as np
import pandas as pd
from transformers import ASTForAudioClassification, Trainer, TrainingArguments

from dataset import load_dataset_esc50


def squeeze_func(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove the extra dimension from input_values.
    """
    batch['input_values'] = np.asarray(batch['input_values']).squeeze(0)
    return batch


def prepare_datasets(apply_augmentation: bool) -> Tuple[Any, List[int]]:
    """
    Load and preprocess the ESC-50 dataset.
    """
    dataset, folds = load_dataset_esc50(apply_augmentation=apply_augmentation)
    return dataset.map(squeeze_func), folds


def get_model(model_name: str) -> ASTForAudioClassification:
    """
    Load the AST audio classification model with SDPA.
    """
    return ASTForAudioClassification.from_pretrained(
        model_name,
        attn_implementation="sdpa"
    )


def make_trainer(
    model: ASTForAudioClassification,
    train_dataset: Any,
    eval_dataset: Any,
    args: TrainingArguments
) -> Trainer:
    """
    Build and return a HuggingFace Trainer instance.
    """
    metric = evaluate.load('accuracy')

    def compute_metrics(pred):
        logits, labels = pred
        return metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )


def save_results(results: List[Dict[str, Any]], csv_path: str) -> None:
    """
    Save the experiment results to a CSV file.
    """
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")