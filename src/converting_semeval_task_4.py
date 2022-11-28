import glob,json
import pandas as pd









def create_dataset_4_sent_classification(a_path):
    all_mentions = list()
    for doc in glob.glob('{}/*.txt'.format(a_path)):
        with open(doc) as f:
            lines = f.readlines()
            name = doc.split('/')[-1][:-4]
            lines = [x.split('\t') for x in lines[1:]]
            lines = [(name,x.split('-')[0],x.split('-')[1],x.split('-')[2])
             for y in lines for x in y[2:]]
            all_mentions.extend(lines)
    df = pd.DataFrame(all_mentions,columns=['entity'])

    return df

def create_dataset_4_event_detection(a_path):
    l_4_ev = list()
    for doc in glob.glob('{}*'.format(a_path)):
        print(doc)
        df = pd.read_csv(doc)
        df.tag = df.tag.fillna('0')
        df.tag = df.tag.apply(lambda x:'EVENT' if x=='EVENT_MENTION' else x)
        df = df.groupby('sentence').aggregate({'token':list,'tag':list})
        df = df.reset_index()
        sents = [x[-2] for x in df.iloc[:].values if 'EVENT' in x[-1]]
        labels = [x[-1] for x in df.iloc[:].values if 'EVENT' in x[-1]]
        l_4_ev.append({'sentence':sents,'labels':labels})
    return l_4_ev
