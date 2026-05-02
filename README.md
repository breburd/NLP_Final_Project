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

`python -m pytest`
<br>**or**<br> 
`python -m pytest -v`
<br>**or**<br>
`python -m pytest -s`

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

This dataset was created using the maildir.zip dataset including in the repository. Thr following preprocessing was performed
to create the dataset used within training and testing:

- Windows issue encountered: Ran the `rename_files.py` file because the file names ended with a '.' character, which is invalid in Windows OS.
- Ran the `preprocess/preprocess.py` file that cleaned the dataset and performed weak labeling.

### (Optional) Command Line Arguments
`--data_path`: The path to the directory to the dataset

`--train_path`, `--valid_path`, `--test_path`: Optional paths to pre-made split CSV files. If all three are provided, the model scripts will use those files directly instead of re-splitting the full dataset.

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

For the fixed-split workflow, first create the deterministic split files once:

```bash
python split_dataset.py --data_path preprocess/enron_emails_labeled.csv --output_dir preprocess/splits
```

This writes:

- `preprocess/splits/train.csv`
- `preprocess/splits/valid.csv`
- `preprocess/splits/test.csv`

You can then run a baseline directly with those saved splits:

```bash
python models/logistic_regression.py --train_path preprocess/splits/train.csv --valid_path preprocess/splits/valid.csv --test_path preprocess/splits/test.csv
```

Or run the orchestration script with the same fixed split files:

```bash
python run_baselines.py --train_path preprocess/splits/train.csv --valid_path preprocess/splits/valid.csv --test_path preprocess/splits/test.csv --run_keyword
```

The Colab notebook `runner_10_experiments.ipynb` now follows this workflow too: it creates the splits once, then reuses the same `train`, `valid`, and `test` files across all 10 BERT experiments.

## Make Predictions
The trained models created from our experiments have been provided for testing the predictions 
made. The `test_emails.json` file has been provided as an example for how the model expects
the new data to be fed in. Try out the models yourself by running the following command:
`python models/make_prediction.py --model_name=<model_name> --email_json_path=test_emails.json`.

The following model names can be used:
- bert_baseline_default
- bert_baseline_lr5e-6
- bert_baseline_train10k
- bert_baseline_train45k
- roberta_baseline
