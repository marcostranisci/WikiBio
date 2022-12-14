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

training_sets = [('misc+wiki_4_events_train','wiki_bio_4_event_dev_half')]
for tr_set in training_sets:
    model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
        num_classes=2)

    with open('../data/{}.json'.format(tr_set[0])) as f:
        jsn = json.load(f)


    train_txt = [x[0] for x in jsn['sentences']]
    train_labels = [x[1] for x in jsn['sentences']]
    train_txt = shuffle(train_txt,random_state=r)
    train_labels = shuffle(train_labels,random_state=r)
    print(train_txt[0])

    with open('../data/{}.json'.format(tr_set[1])) as f:
        jsn = json.load(f)
    dev_txt = [x[0] for x in jsn['sentences']]
    dev_labels = [x[1] for x in jsn['sentences']]
    dev_txt = shuffle(dev_txt,random_state=r)
    dev_labels = shuffle(dev_labels,random_state=r)
    print(dev_txt[0])

    validate = data.vanilla_dataset("distilbert-base-uncased",dev_txt[:1123],dev_labels[:1123])
    eval_ = DataLoader(validate,batch_size=10,shuffle=False)





    with open('../data/wiki_bio_4_event_test.json') as f:
        jsn = json.load(f)

    texts = [x[0] for x in jsn['sentences']]
    labels = [x[1] for x in jsn['sentences']]
    texts = shuffle(texts,random_state=r)
    labels = shuffle(labels,random_state=r)

    test = data.vanilla_dataset("distilbert-base-uncased",texts,labels)
    test_ = DataLoader(test,batch_size=10,shuffle=False)

    '''for t in texts[:int(len(texts)/3)]:
        train_txt.append(t)
    for l in labels[]:int(len(labels)/3)]:
        train_labels.append(l)
    #train_labels.extend(texts[:int(len(texts)/3)])'''




    '''if len(train_txt)>=len(validate)*4:
        train = data.vanilla_dataset("distilbert-base-uncased",train_txt[:len(validate)*4],train_labels[:len(validate)*4])'''
    train = data.vanilla_dataset("distilbert-base-uncased",train_txt,train_labels)
    training = DataLoader(train,batch_size=10,shuffle=True)

    tr = Trainer(model,training,eval_,test_,'DEF_{}_{}_report_{}.csv'.format(tr_set[0],tr_set[1],r))
    tr.do_train()

'''
with open('../data/wiki_bio_4_event.json') as f:
    jsn = json.load(f)

train_txt = [x['sentence'] for x in jsn]
train_labels = [x['labels'] for x in jsn]
train_txt = [x for y in texts for x in y]
train_labels = [x for y in labels for x in y]
shuffle(train_txt,random_state=r)
shuffle(train_labels,random_state=r)

eval_sets = ['onto_4_event','timebank_4_event','timelines_4_event','litbank_4_event']
for ev in eval_sents:
    model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
        num_classes=2)
    texts = [x['sentence'] for x in jsn]
    labels = [x['labels'] for x in jsn]
    texts = [x for y in texts for x in y]
    labels = [x for y in labels for x in y]
    shuffle(train_txt,random_state=r)
    shuffle(train_labels,random_state=r)

    train = data.vanilla_dataset("distilbert-base-uncased",train_txt,train_labels)
    training = DataLoader(train,batch_size=10,shuffle=True)
    validate = data.vanilla_dataset("distilbert-base-uncased",texts[:len(train)],labels)
    eval_ = DataLoader(validate,batch_size=10,shuffle=False)
    tr = Trainer(model,training,eval_,'wikibio_report_{}_{}.csv'.format(ev,r))
    tr.do_train()
'''
