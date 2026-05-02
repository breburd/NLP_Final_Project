import os
from xml.parsers.expat import model

from envs.en605645.Lib import email
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import argparse
from pathlib import Path
import pandas as pd

from common import DEFAULT_DATA_PATH, PROJECT_ROOT
from bert_baseline import EnronDataset, MyTrainer, create_explanations

DEFAULT_MODELS_DIR = PROJECT_ROOT / "pretrained_models"

if __name__ == "__main__":
    """
    Load a pretrained model from one of the experiments and use it to make a prediction on an input email.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_models_dir", type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--email_json_path", type=str)  # Path to the JSON file containing the email(s) to be predicted on
    args = parser.parse_args()

    # Load in the saved model and tokenizer
    trained_model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_models_dir / args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_models_dir / args.model_name)

    exp_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    exp_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("Moving model to device ..." + str(args.device))
    trained_model.to(args.device)
    exp_model.to(args.device)

    # Read in the email data into a dataset and make a prediction using the loaded model on the email
    email_df = pd.read_json(args.email_json_path)  # Assuming the JSON file has the same structure as the original CSV file used for training
    data = EnronDataset(
        email_df['from'],
        email_df['to'],
        email_df['subject'],
        email_df['body'],
        privileged=[0]*len(email_df),  # To be predicted, so we can set this to False or any value
        tokenizer=tokenizer,
        max_len=512
    )

    trained_model.eval()

    predictions = []

    for i in range(len(data)):
        item = data[i]  # this should return tokenized tensors

        input_ids = item["input_ids"].unsqueeze(0).to(args.device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(args.device)

        with torch.no_grad():
            output = trained_model(input_ids=input_ids, attention_mask=attention_mask)

        logits = output.logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

    explanations = create_explanations(data.email, predictions, exp_model, exp_tokenizer)

    for i, row in email_df.iterrows():
        print(f"Input: From: {row['from']}, To: {row['to']}, Subject: {row['subject']}, Body: {row['body']}")
        print(f"Prediction: {'Not Privileged' if predictions[i] == 0 else 'Privileged'} because {explanations[i]}\n")
