python BERT-NER-CHINESE \
    --do_eval \
    --predict_file BERT-NER-CHINESE/data/cner/test.char.bmes \
    --device cuda \
    --model_name_or_path bert-base-chinese \
    --load_model_path BERT-NER-CHINESE/log/output/20200906_14_57 \
    --load_entity_label_path BERT-NER-CHINESE/data/cner/ \
    --batch_size 16 \
    --output_dir BERT-NER-CHINESE/log/output 