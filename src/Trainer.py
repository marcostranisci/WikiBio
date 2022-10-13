from pathlib import Path
import data,preproc
from preproc import MultiDoc4seq2seq
import numpy as np
import torch
from model import Bert4EventExtraction
from tqdm import tqdm
import logging as log
import json,os,csv,time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from typing import Dict,List

log.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=log.INFO,
    datefmt='%d/%m/%Y %H:%M:%S'
)

class EarlyStopping():
    def __init__(self, tolerance=2, min_delta=0.05):

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

class Report():
    def __init__(self,file,fields:List=None):
        self.file = file
        self.fields = fields

    def make_report(self,values:Dict):
        if os.path.exists('./report/{}'.format(self.file)):
            with open('./report/{}'.format(self.file),mode='a') as f:
                writer = csv.DictWriter(f,self.fields)
                writer.writerow(values)

        else:
            with open('./report/{}'.format(self.file),mode='a') as f:
                writer = csv.DictWriter(f,self.fields)
                writer.writeheader()
                writer.writerow(values)


        f.close()



class Trainer():
    def __init__(self,model,train,eval,report,EPOCHS=10,LEARNING_RATE=1e-5):

        self.model = model
        self.train = train
        self.eval = eval
        self.report = report
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE






    def do_train(self):
        rep = Report(self.report)
        rep.fields = ['time','epoch','type','loss','accuracy','precision','recall','f_score']


        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.LEARNING_RATE)

        if use_cuda:
            model = model.cuda()
        log.info(f"we are using {device}")

        bench_loss = 10




        early_stopping = EarlyStopping()
        for epoch_num in range(self.EPOCHS):

            total_acc_train = 0
            total_loss_train = 0
            total_prec_train = 0
            total_rec_train = 0
            total_fscore_train = 0

            self.model.train()


            for i,batch in enumerate(tqdm(self.train,total=len(self.train))):


                    train_label = batch['targets'].to(device)

                    mask = batch['attention_mask'].squeeze(1).to(device)
                    input_id = batch['features'].squeeze(1).to(device)

                    optimizer.zero_grad()
                    loss, logits = self.model(features=input_id, attention_mask=mask,labels=train_label)



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

            log.info(
                f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(self.train): .3f} | Accuracy: {total_acc_train / len(self.train): .3f} | Precision: {total_prec_train / len(self.train): .3f} | Recall: {total_rec_train / len(self.train): .3f} | F-Score: {total_fscore_train / len(self.train): .3f}"
                )

            rep.make_report(
            {'time':time.time(),
            'epoch':epoch_num+1,
            'type' : 'train',
            'loss' : total_loss_train / len(self.train),
            'accuracy':total_acc_train / len(self.train),
            'precision' : total_prec_train / len(self.train),
            'recall':total_rec_train / len(self.train),
            'f_score' : total_fscore_train / len(self.train),

            }
            )

            self.model.eval()

            total_acc_val = 0
            total_loss_val = 0
            total_prec_val = 0
            total_rec_val = 0
            total_fscore_val = 0

            for i,batch in enumerate(tqdm(self.eval,total=len(self.eval))):

                val_label = batch['targets'].to(device)
                mask = batch['attention_mask'].squeeze(1).to(device)
                input_id = batch['features'].squeeze(1).to(device)

                with torch.no_grad():
                    loss, logits = self.model(features=input_id, attention_mask=mask,labels=val_label)

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


                log.info(
                f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / len(self.eval): .3f} | Accuracy: {total_acc_val / len(self.eval): .3f} | Precision: {total_prec_val / len(self.eval): .3f} | Recall: {total_rec_val / len(self.eval): .3f} | F-Score: {total_fscore_val / len(self.eval): .3f}"
                )

                rep.make_report(
                {'time':time.time(),
                'epoch':epoch_num+1,
                'type' : 'eval',
                'loss' : total_loss_val / len(self.eval),
                'accuracy': total_acc_val / len(self.eval),
                'precision' : total_prec_val / len(self.eval),
                'recall': total_rec_val / len(self.eval),
                'f_score' : total_fscore_val / len(self.eval),

                }
                )


                early_stopping(total_loss_train/len(self.train),total_loss_val/len(self.eval))
                log.info(f"the delta between training and evaluation is {early_stopping.min_delta}")
                torch.save(self.model.state_dict(),'./fine_tuning_output/boh.pth')
                if early_stopping.early_stop:

                    break


            if total_loss_val / len(self.eval) < bench_loss:
                torch.save(self.model.state_dict(),'./fine_tuning_output/boh.pth')
                bench_loss = total_loss_val / len(self.eval)





            #f.write(f"Epochs: {epoch_num + 1} | Val_Loss: {total_loss_val / loader[idx]['val_length']: .3f} | Accuracy: {total_acc_val / loader[idx]['val_length']: .3f} | Precision: {total_prec_val / loader[idx]['val_length']: .3f} | Recall: {total_rec_val / loader[idx]['val_length']: .3f} | F-Score: {total_fscore_val / loader[idx]['val_length']: .3f}\n")



LEARNING_RATE = 1e-5
EPOCHS = 10
