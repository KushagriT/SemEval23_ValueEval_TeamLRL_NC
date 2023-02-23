import pandas as pd
from transformers import RobertaTokenizerFast
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import Dataset
import torch.nn as nn
from transformers import (
                          RobertaForSequenceClassification,
                          TrainingArguments, Trainer)
import torch
from sklearn.metrics import f1_score
import numpy as np
import evaluate
import random

df_val = pd.read_csv('/content/arguments-validation.tsv',sep='\t')
df_train = pd.read_csv('/content/arguments-training.tsv',sep='\t')
df_val_labels = pd.read_csv('/content/labels-validation.tsv',sep='\t')
df_train_labels = pd.read_csv('/content/labels-training.tsv',sep='\t')
df_val_level1 = pd.read_csv('/content/level1-labels-validation.tsv',sep='\t')
df_train_level1 = pd.read_csv('/content/level1-labels-training.tsv',sep='\t')

df_val['Stance'] = df_val['Stance'].replace({'in favour of':'in favor of'})

df_train['Stance'] = df_train['Stance'].replace({'in favour of':'in favor of'})

df_val_comp_aux = pd.DataFrame()
df_train_comp_aux = pd.DataFrame()

df_val_comp_aux['text'] = df_val ['Conclusion'] + ' </s> '  + df_val['Premise']
df_val_comp_aux ['aux_label'] = df_val['Stance'].replace({'in favor of':0,'against':1})
df_val_comp_aux  ['label'] = df_val_labels[df_val_labels.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)
df_val_comp_aux ['level1'] = df_val_level1[df_val_level1.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)

sort_ind_val = df_val_comp_aux.text.str.len().sort_values().index
df_val_comp_aux = df_val_comp_aux.reindex(sort_ind_val)

df_train_comp_aux['text'] = df_train ['Conclusion'] + ' </s> '  + df_train['Premise']
df_train_comp_aux ['aux_label'] = df_train['Stance'].replace({'in favor of':0,'against':1})
df_train_comp_aux ['label'] = df_train_labels[df_train_labels.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)
df_train_comp_aux ['level1'] = df_train_level1[df_train_level1.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)

sort_ind_train = df_train_comp_aux.text.str.len().sort_values().index
df_train_comp_aux = df_train_comp_aux.reindex(sort_ind_train)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

df_val_comp_aux.reset_index(inplace=True)
df_train_comp_aux.reset_index(inplace=True)

df_train_comp_aux = pd.concat([df_train_comp_aux,df_val_comp_aux],axis=0)

df_val = pd.read_csv('/content/arguments-validation-zhihu.tsv',sep='\t')
df_val_labels = pd.read_csv('/content/labels-validation-zhihu.tsv',sep='\t')
df_val_level1 = pd.read_csv('/content/level1-labels-validation-zhihu.tsv',sep='\t')

df_val['Stance'] = df_val['Stance'].replace({'in favour of':'in favor of'})

df_val_comp_aux = pd.DataFrame()

df_val_comp_aux['text'] = df_val ['Conclusion'] + ' </s> '  + df_val['Premise']
df_val_comp_aux ['aux_label'] = df_val['Stance'].replace({'in favor of':0,'against':1})
df_val_comp_aux  ['label'] = df_val_labels[df_val_labels.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)
df_val_comp_aux ['level1'] = df_val_level1[df_val_level1.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)

sort_ind_val = df_val_comp_aux.text.str.len().sort_values().index
df_val_comp_aux = df_val_comp_aux.reindex(sort_ind_val)

df_val_comp_aux.reset_index(inplace=True)

MAX_LENGTH = 256
BATCH_SIZE = 8
EMBEDDING_SIZE = 768
NUM_LABELS_LEVEL1 = 54
learning_rate_level1 = 5e-4
learning_rate_AUX = 5e-4
epochs_level1 = 25
epochs_AUX = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dtst_train_ = Dataset.from_pandas(df_train_comp_aux)
dtst_val_ = Dataset.from_pandas(df_val_comp_aux)
def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text,  add_special_tokens=True,return_tensors='pt',padding='longest',max_length = MAX_LENGTH)
  encoding["label"] = torch.from_numpy(np.array(examples['aux_label']))
  return encoding

val_aux_dataset = dtst_val_.map(preprocess_data,batched=True,batch_size=BATCH_SIZE)
train_aux_dataset = dtst_train_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text,  add_special_tokens=True,return_tensors='pt',padding='longest',max_length = MAX_LENGTH)
  encoding["label"] = torch.from_numpy(np.array(examples['label']))
  return encoding

val_label_dataset = dtst_val_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)
train_label_dataset = dtst_train_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text,  add_special_tokens=True,return_tensors='pt',padding='longest',max_length = MAX_LENGTH)
  encoding["label"] = torch.from_numpy(np.array(examples['level1']))
  return encoding

val_level1_dataset = dtst_val_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)
train_level1_dataset = dtst_train_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)

level1_labels = list(df_train_level1.columns[1:])
labels_labels = list(df_train_labels.columns[1:])

f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    print(labels)
    return {'f1':f1_score(labels,predictions)}

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(10) 

model_aux = RobertaForSequenceClassification.from_pretrained('/roberta_valueval_pretrain',
                                                                num_labels=2,
                                                                problem_type="single_label_classification",
                                                                classifier_dropout=0)
model_aux.to(device)
for param in model_aux.roberta.parameters():
      param.requires_grad=False
modules = [model_aux.roberta.encoder.layer[10], model_aux.roberta.encoder.layer[11]] 
for module in modules:
  for param in module.parameters():
    param.requires_grad = True
    
args = TrainingArguments(
        output_dir='/content/valueeval_model1_aux',
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate_AUX,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs_AUX,
        weight_decay=0.01,
        optim = 'adamw_torch'
    )
multi_trainer_aux = Trainer(
        model_aux,
        args,
        train_dataset=train_aux_dataset,
        eval_dataset=val_aux_dataset,
        compute_metrics= compute_metrics,
        tokenizer=tokenizer

    )

multi_trainer_aux.train()

multi_trainer_aux.save_model('/valueeval_model_stance')

set_seed(10) 


def accuracy_thresh(y_pred, y_true, thresh=0.5):
    """Compute accuracy of predictions"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    y_pred = nn.Sigmoid()(y_pred)

    return ((y_pred >= thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, value_classes, thresh=0.5):
    """Compute label-wise and averaged F1-scores"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    y_pred = nn.Sigmoid()(y_pred)
    y_true = y_true.bool().numpy()
    y_pred = (y_pred >= thresh).numpy()

    f1_scores = {}
    for i, v in enumerate(value_classes):
        f1_scores[v] = round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 2)

    f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 2)

    return f1_scores


def compute_metrics(eval_pred, value_classes):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, value_classes)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels), 'f1-score': f1scores,
            'marco-avg-f1score': f1scores['avg-f1-score']}


class MultiLabelTrainer(Trainer):
    """
        A transformers `Trainer` with custom loss computation
        Methods
        -------
        compute_loss(model, inputs, return_outputs=False):
            Overrides loss computation from Trainer class
        """
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

model_level1 = RobertaForSequenceClassification.from_pretrained('/roberta_valueval_pretrain',
                                                                num_labels=NUM_LABELS_LEVEL1,
                                                                problem_type="multi_label_classification",
                                                                classifier_dropout=0)
model_level1.to(device)
for param in model_level1.roberta.parameters():
      param.requires_grad=False
modules = [model_level1.roberta.encoder.layer[10], model_level1.roberta.encoder.layer[11]] 
for module in modules:
  for param in module.parameters():
    param.requires_grad = True
    
args = TrainingArguments(
        output_dir='/valueeval_model1_level',
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate_level1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs_level1,
        weight_decay=0.01,
        optim = 'adamw_torch'

    )

multi_trainer_level1 = MultiLabelTrainer(
        model_level1,
        args,
        train_dataset=train_level1_dataset,
        eval_dataset=val_level1_dataset,
        compute_metrics=lambda x: compute_metrics(x, level1_labels),
        tokenizer=tokenizer

    )

multi_trainer_level1.train()

multi_trainer_level1.save_model('/valueeval_model_level1')
