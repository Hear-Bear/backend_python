from transformers import TrainingArguments

from functions import prepare_datasets, make_trainer, get_model


def main():
    output_dir = "output/augmented_esc50"

    test_ds, _ = prepare_datasets(apply_augmentation=False)

    # Use same model & args as training
    model = get_model("MIT/ast-finetuned-audioset-10-10-0.4593")

    training_args = TrainingArguments(output_dir=output_dir)
    trainer = make_trainer(model, None, None, training_args)

    trainer.model = model.from_pretrained(output_dir)

    pred_output = trainer.predict(test_ds)
    trainer.save_metrics("test", pred_output.metrics)

if __name__ == "__main__":
    main()