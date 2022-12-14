from pathlib import Path
import data,preproc
from preproc import MultiDoc4seq2seq
import numpy as np
from model import Bert4EventExtraction
from tqdm import tqdm
import logging as log
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import os,torch,json
from Trainer import Trainer
from sklearn.utils import shuffle


r=2
torch.manual_seed(r)
torch.autograd.set_detect_anomaly(True)

def collate_batch(batch,max_size=50):
    if len(batch['features'])>50:
    features = [batch['features'][i:i+max_size] for i in range(0,len(batch['features']),max_size)]
    targets = [batch['targets'][i:i+max_size] for i in range(0,len(batch['targets']),max_size)]
    attention_mask = [batch['attention_mask'][i:i+max_size] for i in range(0,len(batch['attention_mask']),max_size)]
    for i in range(features):
        splitted_batch = list()

        splitted_batch.append({'features':features[i],'targets':targets:[i],'attention_mask':attention_mask[i]})


    return splitted_batch




training_sets = ['onto_def_4_entities','gum_4_entities']
for tr_set in training_sets:
    model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
        num_classes=2)

    with open('../data/{}.json'.format(tr_set)) as f:
        jsn = json.load(f)

    print(list(jsn[0]))
    train_txt = [x['tokens'] for x in jsn]
    train_labels = [x['labels'] for x in jsn]
    train_txt = shuffle(train_txt,random_state=r)
    train_labels = shuffle(train_labels,random_state=r)

    with open('../data/wiki_bio_4_entities.json') as f:
        jsn = json.load(f)

    texts = [x['tokens'] for x in jsn]
    labels = [x['labels'] for x in jsn]
    texts = shuffle(texts,random_state=r)
    labels = shuffle(labels,random_state=r)


    validate = data.vanilla_dataset("distilbert-base-uncased",texts,labels)
    eval_ = DataLoader(validate,batch_size=1,shuffle=False)
    '''if len(train_txt)>=len(validate)*3:
        train = data.vanilla_dataset("distilbert-base-uncased",train_txt[:len(validate)*3],train_labels[:len(validate)*3])'''
    train = data.vanilla_dataset("distilbert-base-uncased",train_txt,train_labels)
    training = DataLoader(train,batch_size=1,shuffle=True)

    tr = Trainer(model,training,eval_,'{}_report_{}.csv'.format(tr_set,r))
    tr.do_train()
