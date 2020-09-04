  
def set_template(args):

    args.train_file = 'data/cner/dev.char.bmes'
    args.device = 'cuda'
    args.output_dir = ''
    args.model_name_or_path = 'bert-base-chinese'
    args.load_model_path = ''
    args.max_seq_length = 128
    args.batch_size = 4
    args.learning_rate = 3e-5
    args.adam_epsilon = 1e-8
    args.num_train_epochs = 3
    args.warmup_steps = 0