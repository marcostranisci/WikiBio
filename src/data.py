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
            #padding="max_length",
            #max_length=self.max_length,
            return_tensors="pt",
            #truncation=True,
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

        item = output_dict['features'][0]
        feat_splitted = list()
        feat = [list(item[i:i+self.max_length-2]) for i in range(1,len(item),self.max_length-2)]
        for el in feat[:-1]:
            el.append(torch.tensor(102))
            el.insert(0,torch.tensor(101))
            el = torch.stack(el)
            feat_splitted.append(el)
        feat[-1] = feat[-1][:-1]

        for i in range(self.max_length-2-len(feat[-1])):
            feat[-1].append(torch.tensor(100))

        feat[-1].insert(0,torch.tensor(101))
        feat[-1].append(torch.tensor(102))
        feat[-1] = torch.stack(feat[-1])

        feat_splitted.append(feat[-1])
        feat_splitted= torch.stack(feat_splitted)

        item = output_dict['targets']
        tar_splitted = list()
        tar = [list(item[i:i+self.max_length-2]) for i in range(1,len(item),self.max_length-2)]
        for el in tar[:-1]:
            el.append(torch.tensor(102))
            el.insert(0,torch.tensor(101))
            el = torch.stack(el)
            tar_splitted.append(el)
        tar[-1] = tar[-1][:-1]
        for i in range(self.max_length-2-len(tar[-1])):
            tar[-1].append(torch.tensor(100))
        tar[-1].insert(0,torch.tensor(101))
        tar[-1].append(torch.tensor(102))
        tar[-1] = torch.stack(tar[-1])
        tar_splitted.append(tar[-1])

        tar_splitted = torch.stack(tar_splitted)

        item = output_dict['attention_mask'][0]
        mask_splitted = list()
        mask = [list(item[i:i+self.max_length-2]) for i in range(1,len(item),self.max_length-2)]
        for el in mask[:-1]:
            el.append(torch.tensor(102))
            el.insert(0,torch.tensor(101))
            el = torch.stack(el)
            mask_splitted.append(el)
        mask[-1] = mask[-1][:-1]
        for i in range(self.max_length-2-len(mask[-1])):
            mask[-1].append(torch.tensor(100))
        mask[-1].insert(0,torch.tensor(101))
        mask[-1].append(torch.tensor(102))
        mask[-1] = torch.stack(mask[-1])
        mask_splitted.append(mask[-1])
        mask_splitted = torch.stack(mask_splitted)
        output_dict['features'] = feat_splitted
        output_dict['targets'] = tar_splitted
        output_dict['attention_mask'] = mask_splitted

        return output_dict

class MyDataset4Sentence(Dataset):
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


        output_dict = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
            is_split_into_words=True
        )
        tag = torch.Tensor(tag)
        tag = tag.type(torch.LongTensor)
        output_dict['targets'] = tag


        output_dict['features'] = output_dict['input_ids']
        del output_dict['input_ids']
        return output_dict




def vanilla_dataset(model_name,inputs,labels=None,max_length=512):
    my_dataset = MyDataset(model_name=model_name,inputs=inputs, targets=labels,max_length=max_length)

    return my_dataset

def vanilla_dataset4coref(model_name,inputs,labels=None,max_length=128):
    my_dataset = MyDataset4coref(model_name=model_name,inputs=inputs,targets=labels,max_length=max_length)

    return my_dataset
def vanilla_dataset_4_sent(model_name,inputs,labels=None,max_length=256):
    my_dataset = MyDataset4Sentence(model_name=model_name,inputs=inputs, targets=labels,max_length=max_length)

    return my_dataset
