# BERT-NER-CHINESE

使用 BERT 預訓練模型進行中文 NER 任務。

## Dataset

- cner

## Quick Start

### 1. Prepare your training data and install the package in requirement.txt

### 2. Fine-tune BERT model

```
sh tarin.sh
```

<!-- ### 3. Interaction

```
sh interaction.sh
``` -->

### 3. Evaluation

```
sh eval.sh
```

## Experiment

### The experimental result of F1-measure：
```
Eval: 100%|█████████████████████████████████████| 30/30 [00:03<00:00,  8.45it/s]
Average f1 : 0.9666356650437157
```

## Model architectures
BERT (from Google) released with the paper[ BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.


基本上就是這張圖的 fine-tune 情境：

![bert-ner](https://i.imgur.com/5SqW8xs.png)

當兵前無事練習寫的 pytorch BERT-NER 模型，還有許多地方可以加強。