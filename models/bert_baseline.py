import argparse

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from common import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, get_scores, load_data, make_folder, save_json, split_data


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels = list(labels)
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {}
        for key in self.encodings:
            item[key] = torch.tensor(self.encodings[key][index])
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


class MyTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is None:
            loss_function = torch.nn.CrossEntropyLoss()
        else:
            loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

        loss = loss_function(logits.view(-1, model.config.num_labels), labels.view(-1))

        if return_outputs:
            return loss, outputs
        return loss


def maybe_take_some_rows(df, limit):
    if limit is None or limit <= 0:
        return df
    if len(df) <= limit:
        return df
    return df.iloc[:limit].copy()


def metric_function(prediction_output):
    logits, labels = prediction_output
    preds = np.argmax(logits, axis=-1)
    scores = get_scores(labels, preds)
    scores.pop("classification_report", None)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "bert_baseline"))
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--valid_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data_path)
    train_df, valid_df, test_df = split_data(df, seed=args.seed)

    train_df = maybe_take_some_rows(train_df, args.train_size)
    valid_df = maybe_take_some_rows(valid_df, args.valid_size)
    test_df = maybe_take_some_rows(test_df, args.test_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = EmailDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    valid_dataset = EmailDataset(valid_df["text"], valid_df["label"], tokenizer, args.max_length)
    test_dataset = EmailDataset(test_df["text"], test_df["label"], tokenizer, args.max_length)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"],
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    output_dir = make_folder(args.output_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=args.seed,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=metric_function,
        class_weights=class_weights,
    )

    trainer.train()

    valid_output = trainer.predict(valid_dataset)
    test_output = trainer.predict(test_dataset)

    valid_pred = np.argmax(valid_output.predictions, axis=-1)
    test_pred = np.argmax(test_output.predictions, axis=-1)

    results = {
        "baseline": "bert_baseline",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "num_train": len(train_df),
        "num_valid": len(valid_df),
        "num_test": len(test_df),
        "valid": get_scores(valid_df["label"], valid_pred),
        "test": get_scores(test_df["label"], test_pred),
    }

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    save_json(results, output_dir / "metrics.json")

    print("bert baseline finished")
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
