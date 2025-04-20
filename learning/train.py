import os
import argparse
from transformers import TrainingArguments
from functions import (
    prepare_datasets,
    get_model,
    make_trainer,
    save_results
)
from dataset import get_latest_checkpoint, split_by_fold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument('--output_dir', default="./output")
    parser.add_argument('--num_experiments', type=int, default=10)
    parser.add_argument('--csv_path', default="./output/logs_results_augmented_esc50.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data (augmented)
    dataset, _ = prepare_datasets(apply_augmentation=True)

    # Initialize model and Trainer args
    model = get_model(args.model_name)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        do_train=True,
        do_eval=True,
        eval_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy='epoch',
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    results = []
    trainer = None
    # Multiple experiments
    for i in range(args.num_experiments):
        print(f"=== Experiment {i+1} ===")
        train_ds, eval_ds = split_by_fold(dataset)
        trainer = make_trainer(model, train_ds, eval_ds, training_args)

        ckpt = get_latest_checkpoint(args.output_dir)
        if ckpt:
            print(f"Resuming from {ckpt}")
            train_out = trainer.train(resume_from_checkpoint=ckpt)
        else:
            train_out = trainer.train()

        eval_out = trainer.evaluate()
        stats = {f"train_{k}": v for k,v in train_out.metrics.items()}
        stats.update({f"eval_{k}": v for k,v in eval_out.items()})
        stats['experiment_id'] = i+1
        results.append(stats)

    # Save final model (best checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "augmented_esc50"))
    save_results(results, args.csv_path)


if __name__ == "__main__":
    main()