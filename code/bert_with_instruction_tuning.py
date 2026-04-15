# Import libraries
import argparse
import random
import subprocess
from tqdm import tqdm
import json
import evaluate as evaluate
from transformers import get_scheduler
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across random, numpy, and torch libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU

    Source: base_classification.py from Assignment 6
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


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


def evaluate_model(model, dataloader, device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy

    Source: base_classification.py from Assignment 6, but TODO will need to modify it to work 
    with the new dataset and model architecture.
    """

    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # forward pass
        output = model(input_ids, attention_mask)

        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['labels'])

    # compute and return metrics
    return dev_accuracy.compute()


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None

    Source: base_classification.py from Assignment 6, but TODO will need to modify it to work 
    with the new dataset and model architecture.
    """

    # TODO test with multiple learning rates and optimizers -> possibly add command line arguments for
    # the optimizer
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in tqdm(enumerate(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            output = mymodel.forward(input_ids, attention_mask)
            predictions = output['logits']

            # compute the loss using the loss function
            loss_value = loss(predictions, labels)

            # loss backward
            loss_value.backward()

            # update the model parameters with optimizer and lr_scheduler step
            optimizer.step()
            lr_scheduler.step()

            # zero the gradients
            optimizer.zero_grad()

            predictions = torch.argmax(predictions, dim=1)

            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=batch['labels'])
            
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        train_acc = train_accuracy.compute()
        print(f" - Average training metrics: accuracy={train_acc}")
        train_acc_list.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
    # save the training and validation accuracy for each epoch to a json file to build bar plots later
    with open(f'train_acc_{args.model}_{lr}_{num_epochs}.json', 'w', encoding='utf-8') as f:
        json.dump(train_acc_list, f)
    with open(f'dev_acc_{args.model}_{lr}_{num_epochs}.json', 'w', encoding='utf-8') as f:
        json.dump(dev_acc_list, f)

    # generate plots here
    plt.clf()
    plt.plot(epoch_list, train_acc_list, 'b', label='train')
    plt.plot(epoch_list, dev_acc_list, 'g', label='valid')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    save_path = f"base_full_{args.model}_{lr}_{num_epochs}.png"
    plt.savefig(save_path)


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
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, args.device, args.lr)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
