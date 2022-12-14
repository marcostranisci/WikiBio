import glob
import regex as re
from collections import Counter


def modify_gum_labels(a_list,ent):
    gum4entity = list()
    prev = 0
    for line in a_list:

        if re.search('\({}\)'.format(ent),line[-1]):

            gum4entity.append((line[1],'ENTITY'))
            prev = 0

        elif re.search('{}(?!\)|[0-9])'.format(ent),line[-1]):
            gum4entity.append((line[1],'ENTITY'))
            prev = 1
        elif re.search('{}\)'.format(ent),line[-1]):
            gum4entity.append((line[1],'ENTITY'))
            prev = 0
        elif line[-1] == '_\n' and prev==1:
            gum4entity.append((line[1],'ENTITY'))
        elif line[-1] == '_\n' and prev==0:
            gum4entity.append((line[1],'0'))
        else: gum4entity.append((line[1],'0'))
    gum4entity = sentence_identification(gum4entity)
    return gum4entity


def sentence_identification(a_list):
    labelsWsents = list()
    i = 0
    for line in a_list:
        if line[0]=='.':
            labelsWsents.append((line[0],line[1],i))
            i+=1
        else:
            labelsWsents.append((line[0],line[1],i))

    return labelsWsents






def converting_gum_coref(a_path):
    revised_corpus = list()
    for doc in glob.glob(a_path):
        with open(doc) as f:
            lines = f.readlines()
            lines = [x.split('\t') for x in lines[1:-2]]
            ents = [x[-1][:-2] for x in lines if re.search('person|organization',x[-1])]

            c = Counter(ents)
            if len(c)>0:

                ent = c.most_common()[0][0]
                ent = ent[:-1] if ent.endswith(')') else ent
                ent = ent[1:] if ent.startswith('(') else ent
                new_labeled = modify_gum_labels(lines,ent)
                #new_labeled = sentence_identification(new_labeled)
                revised_corpus.append(new_labeled)

    return revised_corpus
