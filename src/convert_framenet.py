import glob
import xml.etree.ElementTree as ET
import pandas as pd





def convert_frames(a_path):
    all_frame_doc = list()
    for doc in glob.glob('{}*.xml'.format(a_path)):
        sents = list()
        labels = list()
        item = ET.parse(doc)
        root = item.getroot()

        for el in root:


            sent = list(el)[0].text.split(' ')
            label = ['0' for x in range(len(sent))]
            for x in list(el)[1:]:
                try:
                    a = x.attrib['status']
                    if a == 'UNANN':
                        tokens = x.getchildren()
                        for t in tokens:
                            if t.attrib['name'] == 'PENN':
                                tok_ids = list()
                                for tok in t.getchildren():
                                    tok_ids.append(tok.attrib['end'])
                    elif a == 'MANUAL':
                        frames = x.getchildren()
                        for f in frames:
                            for fr in f:
                                if fr.attrib['name']=='Target':
                                    frame = fr.attrib['end']
                                    label[tok_ids.index(frame)] = 'EVENT'




                except Exception as e:print(e)
            sents.append(sent)
            labels.append(label)
        all_frame_doc.append({'sentence':sents,'labels':labels})
    return all_frame_doc
