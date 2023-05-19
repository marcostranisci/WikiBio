from pathlib import Path
import data,preproc
from preproc import MultiDoc4seq2seq
import numpy as np
import torch
from model import Bert4EventExtraction
from tqdm import tqdm
import logging as log
import json,os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

log.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=log.INFO,
    datefmt='%d/%m/%Y %H:%M:%S'
)

class EarlyStopping():
    def __init__(self, tolerance=3, min_delta=0.1):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            self.min_delta = validation_loss - train_loss
            if self.counter >= self.tolerance:
                self.early_stop = True

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):

    layer.reset_parameters()



def training(model,loader):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    bench_loss = 10
    for idx in range(5):
        model.apply(reset_weights)
        if idx>=0:

            log.info(f"we are using {device}")

            early_stopping = EarlyStopping()
            for epoch_num in range(EPOCHS):

                total_acc_train = 0
                total_loss_train = 0
                total_prec_train = 0
                total_rec_train = 0
                total_fscore_train = 0

                model.train()


                for i, batch in tqdm(enumerate(loader[idx]['train']),total=len(loader[idx]['train'])):

                    train_label = batch['targets'].to(device)
                    mask = batch['attention_mask'].squeeze(1).to(device)
                    input_id = batch['features'].squeeze(1).to(device)

                    optimizer.zero_grad()
                    loss, logits = model(features=input_id, attention_mask=mask,labels=train_label)

                    for i in range(logits.shape[0]):

                      logits_clean = logits[i][train_label[i] != -100]
                      label_clean = train_label[i][train_label[i] != -100]
                      predictions = logits_clean.argmax(dim=1)

                      acc = (predictions == label_clean).float().mean()
                      total_acc_train += acc
                      total_loss_train += loss.item()

                      precision,recall,fscore,support = precision_recall_fscore_support(label_clean.cpu(),predictions.cpu(),average='macro',zero_division=0)
                      total_prec_train +=precision
                      total_rec_train+=recall
                      total_fscore_train+=fscore

                    loss.backward()
                    optimizer.step()


                model.eval()

                total_acc_val = 0
                total_loss_val = 0
                total_prec_val = 0
                total_rec_val = 0
                total_fscore_val = 0

                for i, batch in tqdm(enumerate(loader[idx]['valid']),total=len(loader[idx]['valid'])):

                    val_label = batch['targets'].to(device)
                    mask = batch['attention_mask'].squeeze(1).to(device)
                    input_id = batch['features'].squeeze(1).to(device)

                    with torch.no_grad():
                        loss, logits = model(features=input_id, attention_mask=mask,labels=val_label)

                    for i in range(logits.shape[0]):

                        logits_clean = logits[i][val_label[i] != -100]
                        label_clean = val_label[i][val_label[i] != -100]


                        predictions = logits_clean.argmax(dim=1)
                        acc = (predictions == label_clean).float().mean()
                        total_acc_val += acc
                        total_loss_val += loss.item()
                        precision,recall,fscore,support = precision_recall_fscore_support(label_clean.cpu(),predictions.cpu(),average='macro',zero_division=0)
                        total_prec_val +=precision
                        total_rec_val+=recall
                        total_fscore_val+=fscore






                print(
                    f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / loader[idx]['train_length']: .3f} | Accuracy: {total_acc_train / loader[idx]['train_length']: .3f} | Precision: {total_prec_train / loader[idx]['train_length']: .3f} | Recall: {total_rec_train / loader[idx]['train_length']: .3f} | F-Score: {total_fscore_train / loader[idx]['train_length']: .3f}"
                    )
                print(
                f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader[idx]['val_length']: .3f} | Accuracy: {total_acc_val / loader[idx]['val_length']: .3f} | Precision: {total_prec_val / loader[idx]['val_length']: .3f} | Recall: {total_rec_val / loader[idx]['val_length']: .3f} | F-Score: {total_fscore_val / loader[idx]['val_length']: .3f}"
                )
                with open('../fine_tuning_output/k_fold_distilbert_report_timebank.txt',mode='a') as f:
                    f.write(f"Fold: {idx} | Epochs: {epoch_num + 1} | Loss: {total_loss_train / loader[idx]['train_length']: .3f} | Accuracy: {total_acc_train / loader[idx]['train_length']: .3f} | Precision: {total_prec_train / loader[idx]['train_length']: .3f} | Recall: {total_rec_train / loader[idx]['train_length']: .3f} | F-Score: {total_fscore_train / loader[idx]['train_length']: .3f}\n")
                    f.write(f"Fold: {idx} | Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader[idx]['val_length']: .3f} | Accuracy: {total_acc_val / loader[idx]['val_length']: .3f} | Precision: {total_prec_val / loader[idx]['val_length']: .3f} | Recall: {total_rec_val / loader[idx]['val_length']: .3f} | F-Score: {total_fscore_val / loader[idx]['val_length']: .3f}\n")

                early_stopping(total_loss_train/loader[idx]['train_length'],total_loss_val/loader[idx]['val_length'])
                log.info(f"the delta between training and evaluation is {early_stopping.min_delta}")
                torch.save(model.state_dict(),'../fine_tuning_output/k_fold_distilbert_report_timebank_tmp.pth')
                if early_stopping.early_stop:

                    break


            if total_loss_val / loader[idx]['val_length'] < bench_loss:
                torch.save(model.state_dict(),'../fine_tuning_output/k_fold_distilbert_report_timebank.pth')
                bench_loss = total_loss_val / loader[idx]['val_length']





            #f.write(f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader[idx]['val_length']: .3f} | Accuracy: {total_acc_val / loader[idx]['val_length']: .3f} | Precision: {total_prec_val / loader[idx]['val_length']: .3f} | Recall: {total_rec_val / loader[idx]['val_length']: .3f} | F-Score: {total_fscore_val / loader[idx]['val_length']: .3f}\n")



LEARNING_RATE = 1e-5
EPOCHS = 10

torch.manual_seed(42)
model = Bert4EventExtraction(pretrained_model_name="distilbert-base-uncased",
    num_classes=2)


texts,labels = preproc.MultiDoc4seq2seq.global_data('./timebank_tabular/',t_type='sents')
with open('./onto_orig.json') as f:
    jsn = json.load(f)

onto_texts = list()
onto_labels = list()
for pair in jsn:
    if len(pair['sentence'])>0:
        onto_texts.append(pair['sentence'])
        lbls = [str(x) for x in pair['labels']]
        onto_labels.append(lbls)
loader = dict()
for r in range(5):
    train_texts,valid_texts= train_test_split(texts,random_state=r)
    train_labels,valid_labels= train_test_split(labels,random_state=r)
    #shuffle(onto_texts,random_state=r)
    #shuffle(onto_labels,random_state=r)
    #train_texts.extend(onto_texts[:len(texts)*4])
    #train_labels.extend(onto_labels[:len(labels)*4])
    train = data.vanilla_dataset("distilbert-base-uncased",train_texts,train_labels)
    valid = data.vanilla_dataset("distilbert-base-uncased",valid_texts,valid_labels)
    loader[r] = {'train':DataLoader(dataset=train,batch_size=16,shuffle=True),
    'valid':DataLoader(dataset=valid,batch_size=16,shuffle=False),
    'train_length':len(train),
    'val_length':len(valid)}



#loader = data.data_for_keyfold("distilbert-base-uncased",texts,labels,n_splits=3)
#test_texts,test_labels = preproc.MultiDoc4seq2seq.global_data('./timebank_test/',t_type='sents')

#test_loader = data.data_for_test("distilbert-base-uncased",test_texts,test_labels)
training(model,loader)
