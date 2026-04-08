# CS 601.471/671: Self-supervised Models
## Final Project: Implementing Intruction-Tuning in 

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
```

### Download the Dataset
TODO -> instructions for how to download the dataset that will be used to run the experiments

### Run the code
TODO -> instructions for running each experiment
