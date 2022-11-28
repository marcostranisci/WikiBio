import glob
import itertools
import regex as re
import logging as log
from collections import Counter

def parse_sents(a_list):
    def_d = dict()

    def_list = list()
    key = '(V*)'
    #def_d['doc'] = a_list[0][0][0]
    for idx in range(len(a_list)-1):
        d = dict()
        d['sentence'] = [w[3] for w in a_list[idx]]
        d['ner'] = [w[10] for w in a_list[idx]]
        d['tree'] = [w[5] for w in a_list[idx]]
        d['entity'] = [w[-2] for w in a_list[idx]]
        '''try:
            a_list[idx] = [w for w in a_list[idx] if len(w)>6]
            d['sentence'] = [w[3] for w in a_list[idx] if len(w)>6]
            d['token_ids'] = [w[2] for w in a_list[idx] if len(w)>6]
            d['lemma'] = [w[6] for w in a_list[idx] if len(w)>6]
            d['pos'] = [w[4] for w in a_list[idx] if len(w)>6]
            d['doc'] = a_list[0][0][0]

            verbs = [(w[6],w.index(key),a_list[idx].index(w)) for w in a_list[idx] if key in w and len(w)>6]
            if len(verbs)>0:
                for verb in verbs:
                    length = len(set([a_list[idx][i][verb[-2]] for i in range(len(a_list[idx])-1)]))
                    if length==2:
                        verbs.remove(verb)
                    else:
                        args = [a_list[idx][i][verb[-2]] for i in range(len(a_list[idx])-1)]
                        d[verb[-1]]=args
                d['verbs'] = [v[0] for v in verbs]
                d['verb_ids'] = [v[-1] for v in verbs]
                id_sent_position_token = [w[:5] for w in a_list[idx]]
                labels = ['0' for i in d['sentence']]

                for v in verbs:
                    labels[v[-1]]='EVENT'
                d['labels'] = labels
            else:
                labels = ['0' for i in d['sentence']]
                d['labels'] = labels
        except Exception as e: print('{},{}'.format(e,a_list[idx][0]))'''

        def_d[idx] = d
        ent = list()
    for item in def_d:
        ents = [re.search('[0-9]+',w).group() for i,w in enumerate(def_d[item]['entity']) if re.search('[0-9]+',w) and (re.search('PERSON',def_d[item]['ner'][i]) or re.search('ORG',def_d[item]['ner'][i]))]
        ent.extend(ents)

        counted = Counter(ent)
    try:
        if len(counted)>0:
            def_d['ent'] = sorted(list(counted))[0]
        else:
            def_d['ent'] = None
    except: def_d






    return def_d



def from_file_to_list(file):
    with open(file) as f:
        all_rows = f.readlines()

    all_rows = [list(b) for a,b in itertools.groupby(all_rows[:],lambda x:x=='\n')]
    all_rows = [x for x in all_rows if len(x)>1]
    all_rows = [[re.sub('\s+','\t',x).split('\t') for x in y] for y in all_rows]
    x = parse_sents(all_rows)
    return x


def multi_doc_conversion(a_path,mod='labels',a_file_name = None):
    conv_d = list()

    for doc in glob.glob(a_path):


        converted = from_file_to_list(doc)

        conv_d.append(converted)

    if mod == 'labels':
        conv_d = modify_onto_labels(conv_d)
    elif mod == 'clusters':
        conv_d = modify_onto_clusters(conv_d)



    return conv_d


def modify_onto_labels(a_list):
    #new_l = list()
    a_list = [x for x in a_list if len(x)>0]
    for el in a_list:
        if el['ent'] is  None:
            continue
        else:
            ent = el['ent']
            for x in el:
                prev = 0
                labels = list()
                try:
                    for i,tok in enumerate(el[x]['entity']):
                        if re.search('\({}$'.format(ent),tok):
                            labels.append('ENTITY')
                            prev = 1
                        elif re.search('\(.*(?<![0-9]){}\)'.format(ent),tok):
                            labels.append('ENTITY')
                            prev = 0
                        elif prev==1 and  el[x]['tree'][i].endswith('*)'):
                            labels.append('ENTITY')
                            prev=0
                        elif tok =='-' and prev == 1 and el[x]['tree'][i]=='*':
                            labels.append('ENTITY')
                        else:
                            labels.append('0')
                            prev=0
                except Exception as e:print(e)
                try:
                    el[x]['labels'] = labels
                except Exception as e:print(e)

    return a_list

def modify_onto_clusters(a_list):
    #new_l = list()
    a_list = [x for x in a_list if len(x)>0]
    for el in a_list:
        if el['ent'] is  None:
            continue
        else:
            ent = el['ent']
            for x in el:
                prev = 0
                labels = list()
                try:
                    for i,tok in enumerate(el[x]['entity']):
                        if re.search('\({}$'.format(ent),tok):
                            labels.append('(ENTITY')
                            prev = 1
                        elif re.search('\(.*(?<![0-9]){}\)'.format(ent),tok):
                            labels.append('(ENTITY)')
                            prev = 0
                        elif prev==1 and  el[x]['tree'][i].endswith('*)'):
                            labels.append('ENTITY)')
                            prev=0
                        elif tok =='-' and prev == 1 and el[x]['tree'][i]=='*':
                            labels.append('0')
                        else:
                            labels.append('0')
                            prev=0
                except Exception as e:print(e)
                try:
                    el[x]['labels'] = labels
                except Exception as e:print(e)

    return a_list


'''
new_l = list()
    ...: for item in jsn:
    ...:     d = dict()
    ...:     tokens = [x for y in item['sentences'] for x in y]
    ...:     sentences = [item['sentences'].index(y) for y in item['sentences']
    ...: for x in y]
    ...:     labels = [x for y in item['labels'] for x in y]
    ...:     d['tokens'] = tokens
    ...:     d['sentences'] = sentences
    ...:     d['labels'] = labels
    ...:     new_l.append(d)
'''
