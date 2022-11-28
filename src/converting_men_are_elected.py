import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

def extracting_token_ids(a_dict):
    token_ids = list()
    for el in a_dict:
        for item in el['triggers']:
            if item['start_token'] == item['end_token']:
                token_ids.append(item['start_token'])
            else:
                toks = [i for i in range(item['start_token'],item['end_token']+1)]
                token_ids.extend(toks)
    return token_ids


def building_labels(a_biography):
    career_evs = eval(a_biography['career_events'])
    career_evs = extracting_token_ids(career_evs)
    life_evs = eval(a_biography['pl_events'])
    life_evs = extracting_token_ids(life_evs)
    name =  a_biography['name_url']
    print(career_evs,life_evs)

    raw_bio = eval(a_biography['person_info'])
    career = [x['info'][0] for x in raw_bio if x['section']=='Career']
    career = career[0] if len(career) > 0 else None
    personal_life = [x['info'][0] for x in raw_bio if x['section']=='Personal Life']
    personal_life = personal_life[0] if len(personal_life)>0 else None

    return None
