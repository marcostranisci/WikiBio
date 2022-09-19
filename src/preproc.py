from typing import Mapping,List,Tuple
import pandas as pd
import glob

class MultiDoc4seq2seq():
    def __init__(self):
        return self

    def count_lengths(a_csv):
        df = pd.read_csv(a_csv)
        counted = df.groupby(['sentence']).idx.apply(list)
        counted = counted.reset_index()
        counted.idx = counted.idx.apply(lambda x:len(x))
        lengths = counted.idx.values

        return lengths

    def global_lengths(a_path):
        l = list()
        for doc in glob.glob('{}/*.csv'.format(a_path)):
            l.extend(MultiDoc4seq2seq.count_lengths(doc))

        if max(l)<=200: max_len = 256
        else: max_len = 512

        return max_len

    def read_data_sent_split(a_file,column_texts='token',column_labels='simple_label'):
        df = pd.read_csv(a_file)
        df[column_texts] = df[column_texts].apply(lambda x:x.strip())
        texts = df.groupby('sentence')[column_texts].apply(list)
        tags = df.groupby('sentence')[column_labels].apply(list)

        return texts,tags

    def read_data_whole_doc(a_file,column_texts='token',column_labels='simple_label'):
        df = pd.read_csv(a_file)
        df[column_texts] = df[column_texts].apply(lambda x:x.strip())
        texts = df[column_texts].tolist()

        tags = df[column_labels].tolist()

        return texts,tags

    def global_data(a_path, t_type:str='sents'):
        sentences = list()
        labels = list()
        print(sentences)
        if t_type=='sents':
            for doc in glob.glob('{}*.csv'.format(a_path)):
                a,b = MultiDoc4seq2seq.read_data_sent_split(doc)
                sentences.extend(a)
                labels.extend(b)
        elif t_type=='doc':
            for doc in glob.glob('{}*.csv'.format(a_path)):
                
                a,b = MultiDoc4seq2seq.read_data_whole_doc(doc)

                sentences.append(a)
                labels.append(b)

        return sentences,labels
