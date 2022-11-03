from typing import Mapping,List,Tuple
import logging,sys
import torch
from torch.utils.data import Dataset, DataLoader,random_split,SubsetRandomSampler
from transformers import AutoTokenizer
import pandas as pd
import regex as re
from sklearn.model_selection import KFold
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self,model_name,inputs:List[List[str]],targets:List[List[str]],tags_dict=None,num_classes=None,max_length=512):
        self.inputs = inputs
        self.max_length = max_length
        self.targets = targets

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tags_dict is not None:
            self.tags_dict = tags_dict
            self.num_classes = len(tags_dict) if num_classes is None else num_classes
        else:
            tgs = list(set([x for y in self.targets for x in y]))
            self.tags_dict= {x:sorted(tgs).index(x) for x in sorted(tgs)}
            self.num_classes = len(tgs) if num_classes is None else num_classes




    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,item):# -> Mapping[str, torch.Tensor]:
        text = self.inputs[item]
        tag = self.targets[item]
        num_classes = self.num_classes

        output_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
            is_split_into_words=True
        )

        parsed_tag = []
        idxs = output_dict.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(output_dict["input_ids"][0])

        for idx in idxs:

            if idx is None:

                parsed_tag.append(-100)
            else:
                parsed_tag.append(self.tags_dict[tag[idx]])

        parsed_tag = torch.Tensor(parsed_tag)
        parsed_tag = parsed_tag.type(torch.LongTensor)
        output_dict['targets'] = parsed_tag

        output_dict['features'] = output_dict['input_ids']
        del output_dict['input_ids']

        return output_dict

class MyDataset4coref(Dataset):
    def __init__(self,model_name,inputs:List[List[str]],targets:List[List[str]],sents:List[List[str]],tags_dict=None,num_classes=None,max_length=512):
        self.inputs = inputs
        self.max_length = max_length
        self.targets = targets
        self.sent_map = sents

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tags_dict is not None:
            self.tags_dict = tags_dict
            self.num_classes = len(tags_dict) if num_classes is None else num_classes
        else:
            tgs = list(set([x for y in self.targets for x in y]))
            self.tags_dict= {x:sorted(tgs).index(x) for x in sorted(tgs)}
            self.num_classes = len(tgs) if num_classes is None else num_classes




    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,item):# -> Mapping[str, torch.Tensor]:
        text = self.inputs[item]
        tag = self.targets[item]
        sent_map = self.sent_map[item]
        num_classes = self.num_classes

        output_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            #padding="max_length",
            #max_length=self.max_length,
            return_tensors="pt",
            #truncation=True,
            return_attention_mask=False,
            is_split_into_words=True
        )

        parsed_tag = []
        parsed_sents = []
        idxs = output_dict.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(output_dict["input_ids"][0])

        for idx in idxs:

            if idx is None:

                parsed_tag.append(-100)
                parsed_sents.append(-100)
            else:
                parsed_tag.append(self.tags_dict[tag[idx]])
                parsed_sents.append(sent_map[idx])

        truncated_sents= list()
        truncated_tokens = list()
        truncated_labels = list()
        truncated_idxs = list()
        while len(parsed_sents)>0:
            if len(parsed_sents)>256:
                n = parsed_sents[255]
                if n == parsed_sents[256]:
                    for i,el in enumerate(parsed_sents[:255]):

                        if el<n-1:
                            bound = i

                    truncated_sents.append(parsed_sents[:bound+1])
                    truncated_tokens.append(tokens[:bound+1])
                    truncated_labels.append(parsed_tag[:bound+1])
                    truncated_idxs.append(idxs[:bound+1])
                    parsed_sents = parsed_sents[bound+1:]
                    tokens = tokens[bound+1:]
                    parsed_tag = parsed_tag[bound+1:]
                    idxs = idxs[bound+1:]
                else:

                    truncated_sents.append(parsed_sents[:n+1])
                    truncated_tokens.append(tokens[:n+1])
                    truncated_labels.append(parsed_tag[:n+1])
                    truncated_idxs.append(idxs[:n+1])
                    parsed_sents = parsed_sents[n+1:]
                    tokens = tokens[n+1:]
                    parsed_tag = parsed_tag[n+1:]
            else:
                truncated_sents.append(parsed_sents)
                truncated_tokens.append(tokens)
                truncated_labels.append(parsed_tag)
                truncated_idxs.append(idxs)
                parsed_sents = []
        for item in truncated_sents:

            if item[0]!=-100: item.insert(0,-100)
            if item[-1]!=-100: item.append(-100)

        for item in truncated_labels:
            if item[0]!=-100: item.insert(0,-100)
            if item[-1]!=-100: item.append(-100)

        for item in truncated_tokens:
            if item[0]!='[CLS]': item.insert(0,'[CLS]')
            if item[-1]!='[SEP]': item.append('[SEP]')

        for item in truncated_idxs:
            if item[0] is not None: item.insert(0,None)
            if item[-1] is not None: item.append(None)

        truncated_sents = [x for y in truncated_sents for x in y]
        truncated_labels = [x for y in truncated_labels for x in y]
        truncated_idxs = [x for y in truncated_idxs for x in y]

        start = None
        clusters = list()
        for i,lab in enumerate(truncated_labels):

            if start is None and lab==1:
                if truncated_labels[i+1]==1:

                    start = truncated_idxs[i]
                    cl = [start]
                else:
                    clusters.append([truncated_idxs[i],truncated_idxs[i]])

            elif start is not None and lab==1:
                if truncated_labels[i+1]==0:

                    end = truncated_idxs[i]
                    cl.append(end)
                    clusters.append(cl)
                    start = None
                else: continue
            '''if lab==1 and prev==0 and truncated_labels[i+1]==0:
                clusters.append([truncated_idxs[i],truncated_idxs[i]])'''



        #parsed_tag = torch.Tensor(parsed_tag)
        #parsed_tag = parsed_tag.type(torch.LongTensor)
        #output_dict['targets'] = truncated_labels
        del output_dict['input_ids']
        #parsed_sents = torch.Tensor(parsed_sents)
        #parsed_sents = parsed_sents.type(torch.LongTensor)
        output_dict['doc_key'] = ""
        output_dict['speakers'] = []
        output_dict['clusters'] = clusters
        output_dict['sentences'] = truncated_tokens
        output_dict['sentence_map'] = truncated_sents
        output_dict['subtoken_map'] = truncated_idxs


        return output_dict


def reading_data(model_name,inputs,targets=None,max_length=512):
    my_dataset = MyDataset(model_name=model_name,inputs=inputs, targets=targets,max_length=max_length)
    train_size = int(0.8 * len(my_dataset))
    test_size = len(my_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(my_dataset,[train_size,test_size])
    train_length = len(train_set)
    val_length = len(val_set)

    my_loader = {'train':DataLoader(dataset=train_set,batch_size=16,shuffle=True),
    'valid':DataLoader(dataset=val_set,batch_size=16,shuffle=False),
    'train_length':train_length,
    'val_length':val_length
    }

    return train_set,val_set

def data_for_keyfold(model_name,inputs,targets=None,max_length=512,n_splits=3):

    my_dataset = MyDataset(model_name=model_name,inputs=inputs, targets=targets,max_length=max_length)
    kfold = KFold(n_splits=n_splits,shuffle=True)
    my_loader = dict()
    for i,(train_ids,val_ids) in enumerate(kfold.split(my_dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_length = len(train_subsampler)
        val_length = len(val_subsampler)

        my_loader[i] = {'train':DataLoader(dataset=my_dataset,batch_size=16,sampler=train_subsampler),
        'valid':DataLoader(dataset=my_dataset,batch_size=16,sampler=val_subsampler),
        'train_length':train_length,
        'val_length':val_length}



    return my_loader

def data_for_test(model_name,inputs,targets=None,max_length=512,batch_size=16):
    my_dataset = MyDataset(model_name=model_name,inputs=inputs, targets=targets,max_length=max_length)
    test_length = len(my_dataset)
    my_test_loader = {'test':DataLoader(dataset=my_dataset,batch_size=n,shuffle=False),
    'length':test_length}

    return my_test_loader

def vanilla_dataset(model_name,inputs,labels=None,max_length=512):
    my_dataset = MyDataset(model_name=model_name,inputs=inputs, targets=labels,max_length=max_length)

    return my_dataset

def vanilla_dataset4coref(model_name,inputs,labels=None,sents=None,max_length=256):
    my_dataset = MyDataset4coref(model_name=model_name,inputs=inputs,targets=labels,sents=sents,max_length=max_length)

    return my_dataset

def stacked_dataset(model_name,inputs:List[Tuple],max_length=512):
    my_dataset = list()
    for texts,labels in tqdm(inputs):
        my_dataset.append(MyDataset(model_name=model_name,inputs=texts,targets=labels))

    return my_dataset
