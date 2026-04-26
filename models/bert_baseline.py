import argparse

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from common import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, get_scores, load_data, make_folder, print_gpu_memory, save_json, split_data, seed_everything


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
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token is not None else "[SEP]"
        input_encoding = f"{self.question} {sep_token} FROM: {self.from_user[index]} TO: {self.to[index]} SUBJECT: {self.subject[index]} EMAIL: {email}"

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
            'labels': torch.tensor(answer, dtype=torch.long),  # labels are the answers (yes/no)
            'text': input_encoding  # we also return the raw text for later use in explanations
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


def generate_explanation(text, label, exp_model, exp_tokenizer):
    """
    Generate an explanation for why a given email was classified as privileged or not privileged. This function takes 
    the raw text of the email and the classification label, and uses the explanation model and tokenizer to generate a
    natural language explanation.
    """
    label_str = "privileged" if label == 1 else "not privileged"

    prompt = f"""
    An email was classified as {label_str}.

    Explain why this classification makes sense based on the content.

    Email:
    {text}

    Explanation:
    """

    inputs = exp_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    # Move inputs to same device as model
    device = exp_model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = exp_model.generate(
        **inputs,
        max_new_tokens=120
    )

    return exp_tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_explanations(texts, labels, exp_model, exp_tokenizer):
    """
    Generate explanations for a list of texts and their corresponding labels using the provided explanation model and tokenizer.
    This function iterates over each text and label, generates an explanation for each pair using the generate_explanation 
    function, and collects the explanations in a list, which is then returned.
    """
    explanations = []
    for text, label in zip(texts, labels):
        explanation = generate_explanation(text, label, exp_model, exp_tokenizer)
        explanations.append(explanation)
    return explanations

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

    exp_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    exp_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("Moving model to device ..." + str(args.device))
    model.to(args.device)
    exp_model.to(args.device)

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
    valid_explanations = create_explanations(valid_dataset.email, valid_dataset.privileged, exp_model, exp_tokenizer)
    test_output = trainer.predict(test_dataset)
    test_explanations = create_explanations(test_dataset.email, test_dataset.privileged, exp_model, exp_tokenizer)

    valid_pred = np.argmax(valid_output.predictions, axis=-1)
    test_pred = np.argmax(test_output.predictions, axis=-1)

    valid_scores = get_scores(valid_df['label'], valid_pred)
    print(f""" - Average DEV metrics: \n
          \tAccuracy: {valid_scores['accuracy']:.4f}\n
          \tPrecision: {valid_scores['precision']:.4f}\n
          \tRecall: {valid_scores['recall']:.4f}\n
          \tF1: {valid_scores['f1']:.4f}
          \tClassification Report: \n{valid_scores['classification_report']}""")
    test_scores = get_scores(test_df['label'], test_pred)
    print(f""" - Average TEST metrics: 
          \tAccuracy: {test_scores['accuracy']:.4f}
          \tPrecision: {test_scores['precision']:.4f}
          \tRecall: {test_scores['recall']:.4f}
          \tF1: {test_scores['f1']:.4f}
          \tClassification Report: \n{test_scores['classification_report']}""")

    results = {
        "baseline": "bert_baseline",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "num_train": len(train_df),
        "num_valid": len(valid_df),
        "num_test": len(test_df),
        "valid": valid_scores,
        "test": test_scores,
        "valid_explanations": valid_explanations,
        "test_explanations": test_explanations,
    }

    print("Saving the model and results...")
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    save_json(results, output_dir / "metrics.json")

    print("bert baseline finished")
    print(output_dir / "metrics.json")
