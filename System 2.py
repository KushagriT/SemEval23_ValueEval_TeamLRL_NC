import numpy as np
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import nltk
import tomotopy as tp
nltk.download('stopwords')
nlp=spacy.load('en_core_web_sm',disable=['parser', 'ner'])
import pandas as pd
from transformers import RobertaTokenizerFast
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import Dataset
import torch.nn as nn
from transformers import (RobertaModel,
                          RobertaForSequenceClassification,
                          TrainingArguments, Trainer)
import torch
from sklearn.metrics import f1_score
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
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

df_val_comp_aux['text'] = 'Arguing ' + df_val['Stance'] + ' ' + df_val ['Conclusion'] + ' by saying '  + df_val['Premise']
df_val_comp_aux  ['label'] = df_val_labels[df_val_labels.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)


sort_ind_val = df_val_comp_aux.text.str.len().sort_values().index
df_val_comp_aux = df_val_comp_aux.reindex(sort_ind_val)

df_train_comp_aux['text'] = 'Arguing ' + df_train['Stance'] + ' ' + df_train ['Conclusion'] + ' by saying '  + df_train['Premise']
df_train_comp_aux ['label'] = df_train_labels[df_train_labels.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)


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

df_val_comp_aux['text'] = 'Arguing ' + df_val['Stance'] + ' ' + df_val ['Conclusion'] + ' by saying '  + df_val['Premise']
df_val_comp_aux  ['label'] = df_val_labels[df_val_labels.columns[1:]].apply(lambda x:np.array(x,dtype=int),axis=1)

sort_ind_val = df_val_comp_aux.text.str.len().sort_values().index
df_val_comp_aux = df_val_comp_aux.reindex(sort_ind_val)

df_val_comp_aux.reset_index(inplace=True)


MAX_LENGTH = 256
BATCH_SIZE = 4
EMBEDDING_SIZE = 768
NUM_LABELS_LABELS = 20
learning_rate_labels = 3e-5
epochs_labels = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

texts_train = df_train_comp_aux['text'].tolist()
texts_val = df_val_comp_aux['text'].tolist()
def tokenize(sentences):

    for sentence in sentences:

         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

stop_words = stopwords.words('english')

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        text_list = []
        for token in doc:
          if token.pos_ in allowed_postags:
            text_list.append(token.lemma_) 
          elif token.pos_=='PROPN':
            text_list.append(token.text) 
        texts_out.append(text_list)
    return texts_out

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

processed_data_train = list(tokenize(texts_train))

data_train = lemmatization(remove_stopwords(processed_data_train))

processed_data_val = list(tokenize(texts_val))

data_val = lemmatization(remove_stopwords(processed_data_val))

term_weight = tp.TermWeight.IDF
ctm = tp.CTModel(tw=term_weight, min_cf=5,k=40, seed=28)

for vec in data_train:
    ctm.add_doc(vec)

ctm.burn_in = 5
ctm.train(0,workers=1)

print('Num docs:', len(ctm.docs), ', Vocab size:', ctm.num_vocabs,
      ', Num words:', ctm.num_words)
print('Removed top words:', ctm.removed_top_words)

for i in range(0, 200, 5):
    ctm.train(5,workers=1) # 100 iterations at a time
    print('Iteration: {}\tLog-likelihood: {}'.format(i, ctm.ll_per_word))
    
    
train_features = []
for id in range(df_train_comp_aux.shape[0]):
    sent_features = ctm.infer(ctm.make_doc(data_train[id]))[0]
    train_features.append(sent_features)

val_features = []
for id in range(df_val_comp_aux.shape[0]):
    sent_features = ctm.infer(ctm.make_doc(data_val[id]))[0]
    val_features.append(sent_features)

ctm_train = train_features
ctm_val = val_features

df_train_comp_aux['ctm'] = ctm_train
df_val_comp_aux['ctm'] = ctm_val


dtst_train_ = Dataset.from_pandas(df_train_comp_aux)
dtst_val_ = Dataset.from_pandas(df_val_comp_aux)

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text,  add_special_tokens=True,return_tensors='pt',padding='longest',max_length = MAX_LENGTH)
  encoding["label"] = torch.from_numpy(np.array(examples['label']))
  encoding['hdp'] = torch.from_numpy(np.array(examples['hdp']))
  return encoding

val_label_dataset = dtst_val_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)
train_label_dataset = dtst_train_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)

labels_labels = list(df_train_labels.columns[1:])


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

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(10) 

class model1(RobertaForSequenceClassification):
  def __init__(self, config, model1=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if model1 is not None:
          self.modellevel1 = model1
          self.dense = nn.Linear(config.hidden_size + self.modellevel1.config.hidden_size + hdp.k, config.hidden_size + self.modellevel1.config.hidden_size + hdp.k)
          self.out_proj = nn.Linear(config.hidden_size + self.modellevel1.config.hidden_size + hdp.k, config.num_labels)
          self.modellevel1.eval()
        else:
          self.modellevel1 = None
          self.dense = nn.Linear(config.hidden_size + ctm.k, config.hidden_size +  hdp.k)
          self.out_proj = nn.Linear(config.hidden_size +  ctm.k, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
          
  def forward(self,input_ids = None,attention_mask = None,labels = None, ctm=None, return_dict= None):
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
    sequence_output = outputs.last_hidden_state[:,0,:]
    if self.modellevel1 is not None:
      
      seq_level1 = self.modellevel1.classifier.dense(self.modellevel1.roberta(input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict).last_hidden_state[:,0,:])
        
      features = torch.cat([sequence_output,seq_level1,ctm],1)
    else:
      features = torch.cat([sequence_output,ctm],1)
    x = features
    x = self.dense(x)
    x = torch.tanh(x)
    logits = self.out_proj(x)

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
          self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
          self.config.problem_type = "single_label_classification"
        else:
          self.config.problem_type = "multi_label_classification"
      if self.config.problem_type == "regression":
        loss_fct = nn.MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model_main = model1.from_pretrained('roberta-base',num_labels=NUM_LABELS_LABELS,
                                   problem_type="multi_label_classification",
                                   model1 = None)
model_main.to(device)
args = TrainingArguments(
        output_dir='/content/valueeval_model1_main',
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate_labels,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=6,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model='marco-avg-f1score',
    )

multi_trainer = MultiLabelTrainer(
        model_main,
        args,
        train_dataset=train_label_dataset,
        eval_dataset=val_label_dataset,
        compute_metrics=lambda x: compute_metrics(x, labels_labels),
        tokenizer=tokenizer

    )

multi_trainer.train()

df_test = pd.read_csv('/arguments-test.tsv',sep='\t')
df_test['Stance'] = df_test['Stance'].replace({'in favour of':'in favor of'})
df_test_comp_aux = pd.DataFrame()
df_test_comp_aux['text'] = 'Arguing ' + df_test['Stance'] + ' ' + df_test ['Conclusion'] + ' by saying '  + df_test['Premise']
df_test_comp_aux['Argument ID'] = df_test['Argument ID'] 
sort_ind_test = df_test_comp_aux.text.str.len().sort_values().index
df_test_comp_aux = df_test_comp_aux.reindex(sort_ind_test)
df_test_comp_aux.reset_index(inplace=True)
texts_test = df_test_comp_aux['text'].tolist()
processed_data_test = list(tokenize(texts_test))

data_test = lemmatization(remove_stopwords(processed_data_test))

test_features = []
for id in range(df_test_comp_aux.shape[0]):
    sent_features = ctm.infer(ctm.make_doc(data_test[id]))[0]
    test_features.append(sent_features)

hdp_test = test_features

df_test_comp_aux['hdp'] = hdp_test

dtst_test_ = Dataset.from_pandas(df_test_comp_aux)

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text,  add_special_tokens=True,return_tensors='pt',padding='longest',max_length = MAX_LENGTH)
  encoding['Argument ID'] = np.array(examples['Argument ID'])
  encoding['ctm'] = torch.from_numpy(np.array(examples['ctm']))
  return encoding

test_label_dataset = dtst_test_.map(preprocess_data, batched=True,batch_size=BATCH_SIZE)

preds = multi_trainer.predict(test_label_dataset)
y_pred = nn.Sigmoid()(torch.from_numpy(preds.predictions))
y_pred = (y_pred >= 0.5).numpy()
y_pred = y_pred.astype(int)
preds_df = pd.DataFrame(y_pred)
preds_df.columns = labels_labels
preds_df['Argument ID'] = test_label_dataset['Argument ID']
cols = list(preds_df.columns)
cols = [cols[-1]] + cols[:-1]
preds_df = preds_df[cols]
preds_df.to_csv('/predictions_arg_test_system2.tsv',sep='\t',index=False)

