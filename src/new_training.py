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


r=1
torch.manual_seed(r)
torch.autograd.set_detect_anomaly(True)

def collate_batch(batch):
    btc = dict()
    feat = [x['features'] for x in batch[0]]
    feat = torch.stack(feat)
    tar = [x['targets'] for x in batch[0]]
    tar = torch.stack(tar)
    mask = [x['attention_mask'] for x in batch[0]]
    mask = torch.stack(mask)
    btc['features'] = feat
    btc['targets'] = tar
    btc['attention_mask'] = mask

    return btc


'''onto_settings = ['orig','mod','mod_jj']
for onto in onto_settings:
    texts,labels = preproc.MultiDoc4seq2seq.global_data('./timebank_tabular/',t_type='sents')
    model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
        num_classes=2)

    with open('./onto_{}.json'.format(onto)) as f:
        jsn = json.load(f)


    train_txt = list()
    train_labels = list()
    for pair in jsn:

        if len(pair['sentence'])>0:
            train_txt.append(pair['sentence'])
            lbls = [str(x) for x in pair['labels']]
            train_labels.append(lbls)

    shuffle(train_txt,random_state=r)
    shuffle(train_labels,random_state=r)
    train = data.vanilla_dataset("distilbert-base-uncased",train_txt[:len(texts)*3],train_labels[:len(texts)*3])
    validate = data.vanilla_dataset("distilbert-base-uncased",texts,labels)

    training = DataLoader(train,batch_size=10,shuffle=True)
    eval = DataLoader(validate,batch_size=10,shuffle=False)


    tr = Trainer(model,training,eval,'{}_report{}_.csv'.format(onto,r))

    tr.do_train()'''

training_sets = ['onto_4_event','timelines_4_event','timebank_4_event']
for tr_set in training_sets:
    model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
        num_classes=2)

    with open('../data/{}.json'.format(tr_set)) as f:
        jsn = json.load(f)

    train_txt = [x['sentence'] for x in jsn]
    train_labels = [x['labels'] for x in jsn]
    train_txt = [x for y in train_txt for x in y]
    train_labels = [x for y in train_labels for x in y]
    shuffle(train_txt,random_state=r)
    shuffle(train_labels,random_state=r)

    with open('../data/wiki_bio_4_event.json') as f:
        jsn = json.load(f)

    texts = [x['sentence'] for x in jsn]
    labels = [x['labels'] for x in jsn]
    texts = [x for y in texts for x in y]
    labels = [x for y in labels for x in y]

    validate = data.vanilla_dataset("distilbert-base-uncased",texts,labels)
    eval = DataLoader(validate,batch_size=10,shuffle=False)

    train = data.vanilla_dataset("distilbert-base-uncased",train_txt[:len(validate)*3],train_labels[:len(validate)*3])
    training = DataLoader(train,batch_size=10,shuffle=True)

    tr = Trainer(model,training,eval,'{}_report_{}.csv'.format(tr_set,r))
    tr.do_train()
