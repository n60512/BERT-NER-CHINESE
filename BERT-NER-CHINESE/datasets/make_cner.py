import torch
import os

def make_cner(loadfile='data/cner/dev.char.bmes', _mode='make'):
    
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

    entities = [val[2] for val in res]
    entities_label = list(set(entities))

    index2entities = dict()
    
    index2entities = {i+1:val for i, val in enumerate(entities_label)}
    
    index2entities[0] = '[PAD]'
    index2entities[len(index2entities)] = '[CLS]'
    index2entities[len(index2entities)] = '[SEP]'
    
    entities2index = {v:i for i, v in index2entities.items()}

    res = [(i, w, entities2index[e]) for i, w, e in res]

    if _mode == 'make':
        return res
    else:
        return index2entities, entities2index