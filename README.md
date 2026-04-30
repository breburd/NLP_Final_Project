# CS 601.471/671: Self-supervised Models
## Final Project: Implementing Intruction-Tuning to Classify Privileged Data with Explanations

<p align="right">
Breanna Burd and Albert Rojas De Jesus
</p>

Encapsulates the NLP Final Project, which includes a proposal, powerpoint presentation, and code that can be used to reproduce the experiments.

### Overview
In this programming final project, we will finetune pretrained LMs for a classification task with privileged/not priveleged answers that produce an explanation for each prediction.
In particular, we will
- Implement Snorkel-style weak labeling on the Enron dataset to imitate a privileged/not privileged dataset.
- Explore various pretrained models (i.e. BERT and GPT via OpenAI).
- Implement instruction-tuning so the predictions include explanations (reasoning for the prediction).

Additionally, we will also experiment with instruction-tuning using BERT and GPT pretrained models via OpenAI APIs

### Setup
create a new environment for this final project:
```
conda create -n nlp_enron_final python=3.10.13
```

And install the required packages:
```
conda activate nlp_enron_final
pip install -r requirements.txt
pip install pytest
```

### Run the Tests
From the project root, run the pytest suite with:
```
python -m pytest -q
```

If you want more detailed test output, run:
```
python -m pytest
`or`
python -m pytest -v
`or`
python -m pytest -s
```

### Download the Dataset
Since privilged data is privileged, there are no existing public datasets that could be used for these experiments.
We created a custom dataset using the Enron dataset that includes emails between individuals and we used weak
labeling to create a target labeled column that defines privileged/not privileged. The saved dataset csv file 
can be found in `preprocess/enron_emails_labeled.csv` and is described as:

```
Number of emails: 517401
Number of users: 64732
Average number of emails per user: 7.99
Number of privileged emails: 50985
Number of non-privileged emails: 466416
Percentage of privileged emails: 9.85%
Columns :  ['from', 'to', 'subject', 'body', 'prob_label', 'label']
```


### (Optional) Command Line Arguments
`--data_path`: The path to the directory to the dataset

`--output_dir`: The path to the output directory for the experiment

`--model_name`: The pretrained model and tokenizer name (see HuggingFace). Default: "bert-base-uncased"

`--max_length`: the maximum length of the encoding. Default: 256

`--epochs`: The number of training epochs. Default: 1

`--train_size`: The number of observations in the training dataset. Default: 30,000

`--valid_size`: The number of observations in the validation dataset. Default: 5,000

`--test_size`: The number of observations in the test dataset. Default: 5,000

`--batch_size`: The batch size. Default: 8

`--learning_rate`: The training learning rate. Default: 2e-5

`--seed`: The random seed. Default: 42

`--device`: The device to run the experiment on. Default: CUDA if available, CPU if not. 

### Run the code
Use Google Colab to leverage the GPUs within a Jupyter Notebook file. The `runner.ipynb` 
includes the experiments we ran for reproducing purposes. This sets up the environment
with th expected versions that can be run together. 
