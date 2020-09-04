from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import torch
from datetime import datetime
from utils.metric import compute_f1

class BertTCTrainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

        if(args.do_train):
            self.train_loader = train_loader
            self.val_loader = val_loader

            self.optimizer = self._create_optimizer()

            # 使用 Adam Optim 更新整個分類模型的參數
            t_total = len(self.train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=args.warmup_steps, 
                num_training_steps = t_total
                )

            self.output_dir= self._create_output_folder()
            self.tb_writer = SummaryWriter('{}/runs'.format(self.output_dir))
            pass

        if(args.do_eval):
            self.test_loader = test_loader
            pass        
        pass

    def train(self):
        
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        best_loss = 100000.0
        running_loss, logging_loss = 0.0, 0.0
        
        global_step = 0
        logging_steps = 50

        for epoch in range(self.args.num_train_epochs):
            
            ep_loss = 0

            for step , batch in enumerate(tqdm(self.train_loader, desc="Epoch", ncols=80)):
                tokens_tensors, segments_tensors, masks_tensors, labels = batch
                tokens_tensors, segments_tensors, masks_tensors, labels = \
                    tokens_tensors.to(self.device), segments_tensors.to(self.device), \
                        masks_tensors.to(self.device), labels.to(self.device)

                # 計算loss
                outputs = self.model(
                    tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors, 
                    labels=labels
                    )

                loss = outputs.loss
                logits = outputs.logits

                loss.backward()

                # 紀錄當前 batch loss
                running_loss += loss.item()
                ep_loss += loss.item()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    self.scheduler.step()  # Update learning rate schedule
                    
                    global_step += 1

                    self.tb_writer.add_scalar('lr', self.scheduler.get_lr()[0], global_step)
                    self.tb_writer.add_scalar(
                        'train loss', 
                        (running_loss - logging_loss)/logging_steps, 
                        global_step
                    )
            
                    logging_loss = running_loss
                    pass
                
                # Validation
                if ((step + 1) % self.args.evaluate_accumulation_steps == 0 and self.args.evaluate_during_training):
                    
                    eval_loss = self._validate()
                    self.tb_writer.add_scalar(
                        'validation loss', 
                        eval_loss, 
                        global_step
                        )
                    
                    if(eval_loss < best_loss):
                        best_loss = eval_loss
                        torch.save(
                            self.model.state_dict(), 
                            '{}/model/best_model.bin'.format(self.output_dir)
                            )
                    
                    self.model.train()
                    pass

                pass
            pass
        pass


    def evaluate(self, tokenizer, _write=False):
        f1_scores = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(self.test_loader, desc='Eval'):
                self.model.eval()
                
                tokens_tensors, segments_tensors, masks_tensors, labels = batch
                tokens_tensors, segments_tensors, masks_tensors, labels = \
                    tokens_tensors.to(self.device), segments_tensors.to(self.device), \
                        masks_tensors.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels
                        )
                    

                    pred_entities = torch.argmax(
                        outputs.logits, 
                        dim=2
                        )
                    stop = 1
                    pass
                    
                f1 = compute_f1(labels[0], pred_entities[0])

                stop = 1

            #     # Compute F1 score
            #     f1 = compute_f1(gold_toks, pred_toks, tokenizer)
            #     f1_scores.append(f1)

            #     # Parm. for simplily display
            #     displayLen = 30
            #     displayStart = best_start-displayLen if best_start-displayLen>0 else 0
            #     displayEnd = best_end+displayLen if best_end+displayLen>0 else 0
            #     pass

            # print('Average f1 : {}'.format(sum(f1_scores)/len(f1_scores)))
        pass

    def interaction(self, tokenizer, qa_text):
        pass


    def _validate(self):
        total_loss = 0.0
        for batch in self.val_loader:
            self.model.eval()
            
            tokens_tensors, segments_tensors, masks_tensors, labels = batch
            tokens_tensors, segments_tensors, masks_tensors, labels = \
                tokens_tensors.to(self.device), segments_tensors.to(self.device), \
                    masks_tensors.to(self.device), labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors, 
                    labels=labels
                    )      
                total_loss += outputs.loss.item()
                pass
        return total_loss/len(self.val_loader)

    def _create_optimizer(self):
        args = self.args
        optimizer = AdamW(
            self.model.parameters(), 
            lr=args.learning_rate, 
            eps=args.adam_epsilon
            )
        return optimizer
    
    def _create_output_folder(self):
        current_time = datetime.now()
        output_dir= '{}/{:%Y%m%d_%H_%M}'.format(self.args.output_dir, current_time)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('{}/model'.format(output_dir), exist_ok=True)
        return output_dir