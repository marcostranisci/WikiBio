import glob
import itertools
import regex as re
import logging as log

def parse_sents(a_list):
    def_d = dict()

    def_list = list()
    key = '(V*)'
    #def_d['doc'] = a_list[0][0][0]
    for idx in range(len(a_list)-1):
        d = dict()

        try:
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
        except Exception as e: print('{},{}'.format(e,a_list[idx][0]))

        def_d[idx] = d
        '''merged= list(zip(id_sent_position_token,labels))
        #print(merged)
        for el in merged:
            del el[0][1]
            el[0].insert(1,idx)
            el[0].append(el[1])
            def_list.append(el[0])
        d['labels'] = labels'''

    return def_d



def from_file_to_list(file):
    with open(file) as f:
        all_rows = f.readlines()

    all_rows = [list(b) for a,b in itertools.groupby(all_rows[1:],lambda x:x=='\n')]
    all_rows = [x for x in all_rows if len(x)>1]
    all_rows = [[re.sub('\s+','\t',x).split('\t') for x in y] for y in all_rows]
    x = parse_sents(all_rows)
    return x


def multi_doc_conversion(a_path,mod = False):
    conv_d = list()

    for doc in glob.glob(a_path):

        converted = from_file_to_list(doc)

        conv_d.append(converted)
    if mod is True:
        conv_d = modify_onto(conv_d)

    return conv_d

def modify_onto(a_list):
    a=0
    b=0
    light_v = ['be','become','make','seem','do','get','have']
    names = [x[:-1] for x in open('../nombank/nombank.1.0.words').readlines()]
    for jsn in a_list:
        for sent in jsn:
            try:
                for i,w in enumerate(jsn[sent]['lemma']):
                    if w in light_v:
                        lemmas = jsn[sent]['lemma'][i:i+4]
                        pos = jsn[sent]['pos'][i:i+4]
                        is_name = False
                        for lemma in lemmas:
                            if lemma in names and re.search('NN',pos[lemmas.index(lemma)]):
                                jsn[sent]['labels'][i]=0
                                jsn[sent]['labels'][i+lemmas.index(lemma)]='EVENT'
                                is_name = True
                                a+=1
                            if is_name is False:
                                for lemma in lemmas:
                                    if re.search('JJ',pos[lemmas.index(lemma)]):
                                        jsn[sent]['labels'][i]=0
                                        jsn[sent]['labels'][i+lemmas.index(lemma)]='EVENT'
                                        b+=1
            except Exception as e:print(f"Houston we've got a {e}")
    print(a,b)
    return a_list
