import pandas as pd
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling
)
from tqdm.auto import tqdm
import torch
import random
from torch.utils.data import Dataset
import numpy as np
from transformers import Trainer, TrainingArguments

df_val = pd.read_csv('/content/arguments-validation.tsv',sep='\t')
df_train = pd.read_csv('/content/arguments-training.tsv',sep='\t')
df_comb = pd.concat([df_train,df_val],axis=0)
df_comb['text'] = df_comb['Conclusion'] + ' </s> ' + df_comb['Stance'] + ' </s> ' + df_comb['Premise']
with open('valueval_text.txt','w') as f:
  for doc in df_comb['text'].tolist():
    f.write(doc)
    f.write('\n')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(10)

class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)
        self.sampler=None
    def load_lines(self,file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines
        
    def encode_lines(self,lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=256
        )

        return batch_encoding["input_ids"]
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)
    
model = RobertaForMaskedLM.from_pretrained('roberta-base')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = MaskedLMDataset('valueval_text.txt',tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir='/content/roberta_ildc_pretrain',
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_gpu_train_batch_size=4,
    save_steps=1000,
    prediction_loss_only=True,
    fp16=True,
    report_to = "wandb"
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

print ('Start a trainer...')
# Start training
trainer.train()
 
# Save
trainer.save_model('/roberta_ildc_pretrain/')
print ('Finished training all...','/roberta_ildc_pretrain')

trainer.save_model('/roberta_valueval_pretrain/')