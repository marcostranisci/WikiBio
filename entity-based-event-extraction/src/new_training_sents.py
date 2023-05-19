from pathlib import Path
import data,preproc
from preproc import MultiDoc4seq2seq
import numpy as np
from model import Bert4EventExtraction,BertForSequenceClassification
from tqdm import tqdm
import logging as log
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import os,torch,json
from Trainer_4_sents import Trainer
from sklearn.utils import shuffle


r=2
torch.manual_seed(r)
torch.autograd.set_detect_anomaly(True)

def collate_batch(batch,max_size=50):
    splitted_batch = list()
    length = len(batch[0]['features'])
    if length>max_size:
        features = [batch[0]['features'][i:i+max_size] for i in range(0,len(batch[0]['features']),max_size)]
        targets = [batch[0]['targets'][i:i+max_size] for i in range(0,len(batch[0]['targets']),max_size)]
        attention_mask = [batch[0]['attention_mask'][i:i+max_size] for i in range(0,len(batch[0]['attention_mask']),max_size)]

        for i in range(len(features)):


            splitted_batch.append({'features':features[i],'targets':targets[i],'attention_mask':attention_mask[i],'length':length})


        return splitted_batch
    else:
        splitted_batch.append({'features':batch[0]['features'],'targets':batch[0]['targets'],'attention_mask':batch[0]['attention_mask'],'length':length})


        return splitted_batch





training_sets = [('onto_4_sentence_detection_all','lorem_4_sentence_detection_dev')\
,('onto_4_sentence_detection_people','onto_4_sentence_detection_dev_people'),('onto_4_sentence_detection_people','lorem_4_sentence_detection_dev'),\
('gum_4_sentence_detection_people','gum_4_sentence_detection_dev_people'),('gum_4_sentence_detection_people','lorem_4_sentence_detection_dev'),\
('gum_4_sentence_detection_all','gum_4_sentence_detection_dev_all'),('gum_4_sentence_detection_all','lorem_4_sentence_detection_dev')]
for tr_set in training_sets:
    model = BertForSequenceClassification(pretrained_model_name="distilbert-base-uncased",num_classes=2)

    with open('../data/sent_labeling/{}.json'.format(tr_set[0])) as f:
        jsn = json.load(f)


    train_txt = [[x for x in y['sentences'][:70]] for y in jsn]
    train_labels = [x['labels'][:70] for x in jsn]
    print(train_txt[0],train_labels[0])
    train_txt = shuffle(train_txt,random_state=r)
    train_labels = shuffle(train_labels,random_state=r)
    if len(train_txt)>200: trn = data.vanilla_dataset_4_sent("distilbert-base-uncased",train_txt[:100],train_labels[:100])
    else: trn = data.vanilla_dataset_4_sent("distilbert-base-uncased",train_txt,train_labels)
    training = DataLoader(trn,batch_size=1,shuffle=False)







    with open('../data/sent_labeling/{}.json'.format(tr_set[1])) as f:
        jsn = json.load(f)
    val_txt = [[x for x in y['sentences']] for y in jsn]
    val_labels = [x['labels'] for x in jsn]
    val_txt = shuffle(val_txt,random_state=r)
    val_labels = shuffle(val_labels,random_state=r)


    validate = data.vanilla_dataset_4_sent("distilbert-base-uncased",val_txt[:5],val_labels[:5])
    eval_ = DataLoader(validate,batch_size=1,shuffle=False)

    with open('../data/sent_labeling/lorem_4_sentence_detection_test.json') as f:
        jsn = json.load(f)
    test_txt = [[x for x in y['sentences']] for y in jsn]
    test_labels = [x['labels'] for x in jsn]
    test_txt = shuffle(test_txt,random_state=r)
    test_labels = shuffle(test_labels,random_state=r)



    tst = data.vanilla_dataset_4_sent("distilbert-base-uncased",test_txt,test_labels)
    test = DataLoader(tst,batch_size=1,shuffle=False)

    tr = Trainer(model,training,eval_,test,'{}_{}_report_{}.csv'.format(tr_set[0],tr_set[1],r))
    tr.do_train()
