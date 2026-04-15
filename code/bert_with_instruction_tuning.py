# Import libraries
import argparse
import random
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across random, numpy, and torch libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EnronDataset(torch.utils.data.Dataset):
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


def load_dataset(model_name):
    """
    Load the dataset from the csv file and return PyTorch DataLoader objects for training, validation, and testing.
    """

    df = pd.read_csv("enron_emails_labeled.csv")

    # Split into train/validation/test -> We want unique users in each split to prevent data leakage.
    # This ensures the model does not learn user-specific patterns that could lead to overfitting
    # and poor generalization on unseen users.
    df['pair_id'] = df.apply(lambda x: "_".join(sorted([x['from'], x['to']])), axis=1)

    # Split once for Train/(Validation + Test)
    gs = GroupShuffleSplit(n_splits=1, train_size=0.7)
    train_idx, temp_idx = next(gs.split(df, groups=df['pair_id']))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Split the remainder into Validation/Test
    gs_val = GroupShuffleSplit(n_splits=1, train_size=0.5)
    val_idx, test_idx = next(gs_val.split(temp_df, groups=temp_df['pair_id']))

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    print("Size of the loaded dataset:")
    print(f" - train: {len(train_df)}")
    print(f" - val: {len(val_df)}")
    print(f" - test: {len(test_df)}")

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loding the data into DS...")
    train_dataset = EnronDataset(
        from_user=list(train_df['from']),
        to=list(train_df['to']),
        subject=list(train_df['subject']),
        email=list(train_df['body']),
        privileged=list(train_df['label']),
        tokenizer=tokenizer,
        max_len=max_len
    )
    validation_dataset = EnronDataset(
        from_user=list(val_df['from']),
        to=list(val_df['to']),
        subject=list(val_df['subject']),
        email=list(val_df['body']),
        privileged=list(val_df['label']),
        tokenizer=tokenizer,
        max_len=max_len
    )
    test_dataset = EnronDataset(
        from_user=list(test_df['from']),
        to=list(test_df['to']),
        subject=list(test_df['subject']),
        email=list(test_df['body']),
        privileged=list(test_df['label']),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return train_dataset, validation_dataset, test_dataset

def pre_process(model_name, batch_size, device):
    train_dataset, validation_dataset, test_dataset = load_dataset(model_name)

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader

if __name__ == "__main__":
    seed_everything()  # Set random seed for reproducibility

    # allow command line arguments for training parameters and model choice
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model, args.batch_size, args.device)

    # TODO train the model, report accuracy, and save the model for inference later.
    # Then we need to build multiple experiments to be run
