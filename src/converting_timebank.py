import glob,json
import pandas as pd


def create_dataset_4_event_detection(a_path):
    l_4_ev = list()
    for doc in glob.glob('{}*.csv'.format(a_path)):
        print(doc)
        df = pd.read_csv(doc)
        df.simple_label = df.simple_label.apply(lambda x:str(x))
        df = df.groupby('sentence').aggregate({'token':list,'simple_label':list})
        df = df.reset_index()
        sents = [x[-2] for x in df.iloc[:].values]
        labels = [x[-1] for x in df.iloc[:].values]
        l_4_ev.append({'sentence':sents,'labels':labels})
    return l_4_ev
