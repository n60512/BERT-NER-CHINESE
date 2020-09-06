import os
import pandas as pd
import pickle

def make_cner(
    loadfile='BERT-NER-CHINESE/data/cner/train.char.bmes', 
    savefile='BERT-NER-CHINESE/data/cner/cner.train.csv', 
    _entfile='BERT-NER-CHINESE/data/cner/'):

    sen_index = 0
    words = list()
    entities = list()
    res = list()

    with open(loadfile, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.split(' ')
            if (len(l)>1):
                w = l[0]
                e = l[1].replace('\n', '')
                words.append(w)
                entities.append(e)
                res.append((sen_index, w, e))
            else:
                sen_index += 1
                pass
        f.close()
        pass

    if(os.path.isfile('{}/index2entities.pickle'.format(_entfile)) and
        os.path.isfile('{}/entities2index.pickle'.format(_entfile)) 
    ):
        print('Entity file exists.')

        with open('{}/index2entities.pickle'.format(_entfile), 'rb') as handle:
            index2entities = pickle.load(handle)      
        with open('{}/entities2index.pickle'.format(_entfile), 'rb') as handle:
            entities2index = pickle.load(handle)        
    else:
        print('Entity file does not exists.\nCreat entity file  ...')

        entities = [val[2] for val in res]
        entities_label = list(set(entities))

        index2entities = dict()
        
        index2entities = {i+1:val for i, val in enumerate(entities_label)}
        
        index2entities[0] = '[PAD]'
        index2entities[len(index2entities)] = '[CLS]'
        index2entities[len(index2entities)] = '[SEP]'
        
        entities2index = {v:i for i, v in index2entities.items()}

        with open('{}/index2entities.pickle'.format(_entfile), 'wb') as handle:
            pickle.dump(index2entities, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('{}/entities2index.pickle'.format(_entfile), 'wb') as handle:
            pickle.dump(entities2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        pass

    res = [(i, w, entities2index[e]) for i, w, e in res]

    _save_as_csv(res, savefile = savefile)

    return res, index2entities, entities2index

def _save_as_csv(res, savefile = 'BERT-NER-CHINESE/data/cner/cner.train.csv'):
    # Define colums
    group = ['sentence', 'entity']
    df = pd.DataFrame(columns = group)

    row_data = {'sen':list(), 'ent':list()}
    idx = 0
    for i, w, e in res:
        if(i==idx):
            row_data['sen'].append(w)
            row_data['ent'].append(e)
        else:
            combined_text = "".join(row_data['sen'])
            combined_entity = ";".join(map(str, row_data['ent']))            
            
            # Appand data into df
            df_row = pd.DataFrame(
                [(combined_text, combined_entity)] , 
                columns = group)
            df = df.append(df_row, ignore_index=True)

            # Initialize next row
            idx+=1
            row_data = {'sen':list(), 'ent':list()}
            row_data['sen'].append(w)
            row_data['ent'].append(e)
            pass
        pass
    df.to_csv(savefile, index = False, header=True)    
    pass

