"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray import tune

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.
train_args=TrainingArguments(
    output_dir='/scratch/cx699',
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=8,
    num_train_epochs=3, 
    evaluation_strategy = "epoch")
## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.
model=finetuning_utils.model_init
compute_metrics=finetuning_utils.compute_metrics
trainer=Trainer(
                args=train_args,
                model_init=model,
                train_dataset=train_data,
                eval_dataset=val_data,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics)
hp_space = lambda x : {"learning_rate": tune.uniform(1e-5,5e-5)}
Opt=trainer.hyperparameter_search(hp_space,search_alg=BayesOptSearch(),n_trials=5,mode='min',compute_objective=lambda compute_metrics:compute_metrics['eval_loss'])
print('Best run ID =',Opt.run_id)
print('Best run Object value =',Opt.objective)
print('Best learning rate =',Opt.hyperparameters)

#extra credit
#random/grid search
from ray.tune.suggest.basic_variant import BasicVariantGenerator
grid=lambda x : {[1e-05,5e-05]}
Best=trainer.hyperparameter_search(grid,search_alg=BasicVariantGenerator(),n_trials=1,mode='min',compute_objective=objective)
print("Best Grid ID : ",Best.run_id)
print("Best Grid Objective : ",Best.objective)
print("Best Grid learning rate : ",Best.hyperparameters)