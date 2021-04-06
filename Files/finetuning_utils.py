from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import RobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    get_score=precision_recall_fscore_support(labels,preds,average='binary')
    accuracy=accuracy_score(labels,preds)
    return {'accuracy score':accuracy,'f1 score':get_score[2],'precision score':get_score[0],'recall score':get_score[1]}

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model=RobertaForSequenceClassification.from_pretrained('roberta-base')
    return model 