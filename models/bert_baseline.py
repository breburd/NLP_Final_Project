import argparse

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from common import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, get_scores, load_data, make_folder, print_gpu_memory, save_json, split_data, seed_everything


# class EmailDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.labels = list(labels)
#         self.encodings = tokenizer(
#             list(texts),
#             truncation=True,
#             padding=True,
#             max_length=max_length,
#         )

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         item = {}
#         for key in self.encodings:
#             item[key] = torch.tensor(self.encodings[key][index])
#         item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
#         return item

class EnronDataset(Dataset):
    """
    Dataset for the dataset of Enron emails with yes/no labels for whether the email is privileged or not.
    """
    def __init__(self, from_user, to, subject, email, privileged, tokenizer, max_len):
        self.from_user = from_user
        self.to = to
        self.subject = subject
        self.email = email
        self.privileged = privileged
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.question = "Is this email privileged? Answer yes or no."

    def __len__(self):
        return len(self.privileged)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        email = str(self.email[index])
        answer = self.privileged[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the email is too long
        input_encoding = f"{self.question} [SEP] FROM: {self.from_user[index]} TO: {self.to[index]} SUBJECT: {self.subject[index]} EMAIL: {email}"

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }
    

class MyTrainer(Trainer):
    """
    Custom Trainer class that allows for class weights to be used in the loss function. This is useful for handling 
    class imbalance in the dataset, where one class may be more prevalent than the other. By providing class weights, 
    we can give more importance to the minority class during training, which can help improve the model's performance on that class.
    """
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
    """
    If a limit is provided and the DataFrame has more rows than the limit, return only the first 'limit' rows of the DataFrame.
    This is useful for quickly testing the model on a smaller subset of the data without having to load the entire dataset."""
    if limit is None or limit <= 0:
        return df
    if len(df) <= limit:
        return df
    return df.iloc[:limit].copy()


def metric_function(prediction_output):
    """
    Compute evaluation metrics for the model's predictions. This function takes the raw output from the model's predictions,
    extracts the logits and true labels, computes the predicted labels by taking the argmax of the logits, and then calculates
    various evaluation metrics such as accuracy, precision, recall, and F1 score using the true labels and predicted labels. 
    The function returns a dictionary of these metrics.
    """
    logits, labels = prediction_output
    preds = np.argmax(logits, axis=-1)
    scores = get_scores(labels, preds)
    scores.pop("classification_report", None)
    return scores


if __name__ == "__main__":
    # Initialize the argument parser and parse the command line arguments for data path, output directory, model name, 
    # max sequence length, number of epochs, sizes of train/validation/test sets, batch size, learning rate, and random seed.
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    print("Loading the data...")
    df = load_data(args.data_path)
    train_df, valid_df, test_df = split_data(df)

    train_df = maybe_take_some_rows(train_df, args.train_size)
    valid_df = maybe_take_some_rows(valid_df, args.valid_size)
    test_df = maybe_take_some_rows(test_df, args.test_size)

    print("Loading the tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    print("Moving model to device ..." + str(args.device))
    model.to(args.device)

    print("Loading the data into PyTorch Datasets...")
    train_dataset = EnronDataset(
        from_user=list(train_df['from']),
        to=list(train_df['to']),
        subject=list(train_df['subject']),
        email=list(train_df['body']),
        privileged=list(train_df['label']),
        tokenizer=tokenizer,
        max_len=args.max_length
    )
    valid_dataset = EnronDataset(
        from_user=list(valid_df['from']),
        to=list(valid_df['to']),
        subject=list(valid_df['subject']),
        email=list(valid_df['body']),
        privileged=list(valid_df['label']),
        tokenizer=tokenizer,
        max_len=args.max_length
    )
    test_dataset = EnronDataset(
        from_user=list(test_df['from']),
        to=list(test_df['to']),
        subject=list(test_df['subject']),
        email=list(test_df['body']),
        privileged=list(test_df['label']),
        tokenizer=tokenizer,
        max_len=args.max_length
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

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
        use_cpu=args.device == "cpu",
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

    print("Starting training...")
    trainer.train()

    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    print("Evaluating the model on the validation and test sets...")
    valid_output = trainer.predict(valid_dataset)
    test_output = trainer.predict(test_dataset)

    valid_pred = np.argmax(valid_output.predictions, axis=-1)
    test_pred = np.argmax(test_output.predictions, axis=-1)

    print(f" - Average DEV metrics: \n{get_scores(valid_df['label'], valid_pred)}")
    print(f" - Average TEST metrics: \n{get_scores(test_df['label'], test_pred)}")

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

    print("Saving the model and results...")
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    save_json(results, output_dir / "metrics.json")

    print("bert baseline finished")
    print(output_dir / "metrics.json")
