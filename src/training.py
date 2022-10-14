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

def evaluate(model,loader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
        log.info('cuda')
    else: log.info('cpu')


    model.eval()

    total_acc_test = 0
    total_loss_test = 0
    total_prec_test = 0
    total_rec_test = 0
    total_fscore_test = 0

    for i, batch in tqdm(enumerate(test_loader['test']),total=len(test_loader['test'])):

        test_label = batch['targets'].to(device)
        mask = batch['attention_mask'].squeeze(1).to(device)
        input_id = batch['features'].squeeze(1).to(device)

        with torch.no_grad():
            loss, logits = model(features=input_id, attention_mask=mask,labels=test_label)

        for i in range(logits.shape[0]):

            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]


            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_test += acc
            total_loss_test += loss.item()
            precision,recall,fscore,support = precision_recall_fscore_support(label_clean.cpu(),predictions.cpu(),average='macro',zero_division=0)
            total_prec_test +=precision
            total_rec_test +=recall
            total_fscore_test +=fscore

    print(f"Test | Test_Loss: {total_loss_test / test_loader['length']: .3f} | Accuracy: {total_acc_test / test_loader['length']: .3f} | Precision: {total_prec_test / test_loader['length']: .3f} | Recall: {total_rec_test / test_loader['length']: .3f} | F-Score: {total_fscore_test / test_loader['length']: .3f}"
        )
    with open('../fine_tuning_output/distilbert_report.txt',mode='a') as f:
            f.write(f"Test | Test_Loss: {total_loss_test / test_loader['length']: .3f} | Accuracy: {total_acc_test / test_loader['length']: .3f} | Precision: {total_prec_test / test_loader['length']: .3f} | Recall: {total_rec_test / test_loader['length']: .3f} | F-Score: {total_fscore_test / test_loader['length']: .3f})\n")

    return total_loss_test / test_loader['length']

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


    log.info(f"we are using {device}")

    early_stopping = EarlyStopping()
    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        total_prec_train = 0
        total_rec_train = 0
        total_fscore_train = 0

        model.train()


        for i, batch in tqdm(enumerate(loader['train']),total=len(loader['train'])):

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

        for i, batch in tqdm(enumerate(loader['valid']),total=len(loader['valid'])):

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
            f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / loader['train_length']: .3f} | Accuracy: {total_acc_train / loader['train_length']: .3f} | Precision: {total_prec_train / loader['train_length']: .3f} | Recall: {total_rec_train / loader['train_length']: .3f} | F-Score: {total_fscore_train / loader['train_length']: .3f}"
            )
        print(
        f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader['val_length']: .3f} | Accuracy: {total_acc_val / loader['val_length']: .3f} | Precision: {total_prec_val / loader['val_length']: .3f} | Recall: {total_rec_val / loader['val_length']: .3f} | F-Score: {total_fscore_val / loader['val_length']: .3f}"
        )
        with open('../fine_tuning_output/distilbert_report_.txt',mode='a') as f:
            f.write(f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / loader['train_length']: .3f} | Accuracy: {total_acc_train / loader['train_length']: .3f} | Precision: {total_prec_train / loader['train_length']: .3f} | Recall: {total_rec_train / loader['train_length']: .3f} | F-Score: {total_fscore_train / loader['train_length']: .3f}\n")
            f.write(f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader['val_length']: .3f} | Accuracy: {total_acc_val / loader['val_length']: .3f} | Precision: {total_prec_val / loader['val_length']: .3f} | Recall: {total_rec_val / loader['val_length']: .3f} | F-Score: {total_fscore_val / loader['val_length']: .3f}\n")

        early_stopping(total_loss_train/loader['train_length'],total_loss_val/loader['val_length'])
        log.info(f"the delta between training and evaluation is {early_stopping.min_delta}")
        torch.save(model.state_dict(),'../fine_tuning_output/distilbert_finetuned_.pth')
        if early_stopping.early_stop: break







            #f.write(f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader['val_length']: .3f} | Accuracy: {total_acc_val / loader['val_length']: .3f} | Precision: {total_prec_val / loader['val_length']: .3f} | Recall: {total_rec_val / loader['val_length']: .3f} | F-Score: {total_fscore_val / loader['val_length']: .3f}\n")



LEARNING_RATE = 1e-5
EPOCHS = 15

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


train = data.vanilla_dataset("distilbert-base-uncased",texts,labels)
test_texts,test_labels = preproc.MultiDoc4seq2seq.global_data('./timebank_test/',t_type='sents')
validate = data.vanilla_dataset("distilbert-base-uncased",test_texts,test_labels)

loader = {'train':DataLoader(dataset=train,batch_size=10,shuffle=True),
'train_length':len(train),'valid':DataLoader(dataset=validate,batch_size=10,shuffle=False),'val_length':len(validate)}
#test_loader = data.data_for_test("distilbert-base-uncased",test_texts,test_labels)
training(model,loader)
