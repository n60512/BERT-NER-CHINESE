from options import args
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
from trainer.bert import BertTCTrainer
from utils.displaydata import displayBertModules
from datasets import dataset_factory
import torch

def train(args):

    train_loader, val_loader = dataset_factory(args, tokenizer)

    print(
        len(train_loader),
        len(val_loader)
    )
    print(args)

    # Load BERT QA pre-trained model
    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, 
        return_dict=True, 
        num_labels=args.tag_nums    # 設定模型 output label nums
        )
    # high-level 顯示此模型裡的 modules
    displayBertModules(model)    

    BertTC = BertTCTrainer(args, model, train_loader, val_loader, None)
    BertTC.train()
    pass

def test(args):

    test_loader = dataset_factory(args, tokenizer)

    # Load BERT QA pre-trained model
    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, 
        return_dict=True, 
        num_labels=args.tag_nums    # 設定模型 output label nums
        )
    model.load_state_dict(
        torch.load('{}/model/best_model.bin'.format(args.load_model_path))
        )
    # high-level 顯示此模型裡的 modules
    displayBertModules(model)
    
    BertTC = BertTCTrainer(args, model, None, None, test_loader)
    BertTC.evaluate(tokenizer)
    pass


if __name__ == "__main__":

    # Get model tokenizer
    PRETRAINED_MODEL_NAME = args.model_name_or_path if args.model_name_or_path!=None else "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    if(args.do_train):
        train(args)
        pass

    if(args.do_eval):
        test(args)
        pass

    pass