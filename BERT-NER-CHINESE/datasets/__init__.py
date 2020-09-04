import torch
from torch.utils.data import DataLoader, random_split
from preprocessing import create_mini_batch
from datasets.datasets import CNERDataset

def dataset_factory(args, tokenizer):

    # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞 
    cner_dataset = CNERDataset(args, tokenizer)
    args.tag_nums = len(cner_dataset.index2entities)

    if(args.do_train):
        train_size = int(0.9 * len(cner_dataset))
        validation_size = len(cner_dataset) - train_size
        
        trainset, validationset = random_split(
            cner_dataset, 
            [train_size, validation_size]
            )

        # 初始化一個每次回傳 64 個訓練樣本的 DataLoader
        # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
        train_loader = DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            collate_fn=create_mini_batch
            )

        val_loader = DataLoader(
            validationset,
            batch_size=args.batch_size, 
            collate_fn=create_mini_batch
        )

        return train_loader, val_loader
    
    if(args.do_eval):
        test_loader = DataLoader(
            cner_dataset, 
            batch_size=args.batch_size, 
            collate_fn=create_mini_batch
            )
        return test_loader