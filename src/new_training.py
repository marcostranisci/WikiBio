from pathlib import Path
import data,preproc
from preproc import MultiDoc4seq2seq
import numpy as np
import torch
from model import Bert4EventExtraction
from tqdm import tqdm
import logging as log
import json
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import os
from Trainer import Trainer




torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)
model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
    num_classes=2)

if os.path.isfile('../fine_tuning_output/distilbert_finetuned_.pth'):
    model.load_state_dict(torch.load('../fine_tuning_output/distilbert_finetuned_.pth'))

texts,labels = preproc.MultiDoc4seq2seq.global_data('./timebank_tabular/',t_type='sents')
with open('./onto_orig.json') as f:
    jsn = json.load(f)



for pair in jsn:

    if len(pair['sentence'])>0:
        texts.append(pair['sentence'])
        lbls = [str(x) for x in pair['labels']]
        labels.append(lbls)

train = data.vanilla_dataset("distilbert-base-uncased",texts[:10],labels[:10])
test_texts,test_labels = preproc.MultiDoc4seq2seq.global_data('./timebank_test/',t_type='sents')
validate = data.vanilla_dataset("distilbert-base-uncased",test_texts[:10],test_labels[:10])
training = DataLoader(dataset=train,batch_size=10,shuffle=True)
eval = DataLoader(dataset=validate,batch_size=10,shuffle=False)
tr = Trainer(model,training,eval,'prova_123.csv')
print(tr)
tr.do_train()
