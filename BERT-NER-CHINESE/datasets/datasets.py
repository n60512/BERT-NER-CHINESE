from torch.utils.data import Dataset
from .make_cner import make_cner
import torch, pandas, os

class CNERDataset(Dataset):
    def __init__(self, args, tokenizer):

        if(args.do_train):
            self.res = make_cner('data/cner/train.char.bmes')
        
        if(args.do_eval):
            self.res = make_cner('data/cner/test.char.bmes')

        self.index2entities, self.entities2index = make_cner(
            'data/cner/train.char.bmes', 
            _mode='r'
            )

        self.len = len(self.res)
        self.tokenizer = tokenizer
    

    def __getitem__(self, idx):

        tokens_sen = list()
        tokens_ent = list()
        for i, w, e in self.res:
            if(i==idx):
                tokens_sen.append(w)
                tokens_ent.append(e)
            pass

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        word_pieces += tokens_sen + ["[SEP]"]
        len_sen = len(word_pieces)

        entity_pieces = [self.entities2index["[CLS]"]]
        entity_pieces += tokens_ent + [self.entities2index["[SEP]"]]
                
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        labels_tensor = torch.tensor(entity_pieces)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_sen, 
                                        dtype=torch.long)


        return (tokens_tensor, segments_tensor, labels_tensor)

    def __len__(self):
        return self.len