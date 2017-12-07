class Const:

    vocab_path = './data_se/word_to_idx.pkl'
    max_words_len_path = './data_se/max_words_len.txt'

    # caption_train_path = '/home/model/work/ai_data/cap_train.json'
    caption_train_vector_path = './data_se/train_vector.pkl'
    # train_image_path = '/home/model/work/ai_data/train'
    # resize_train_out_path = '/home/model/work/ai_data/resize_train'
    resize_train_out_path = '/data/data'

    # val_caption_path = '/home/model/work/ai_data/val/val.json'
    val_vector_out_path = './data_se/val_vector.pkl'
    # val_image_path = '/home/model/work/ai_data/val/val'
    # val_resize_path = '/home/model/work/ai_data/resize_val'
    val_resize_path = '/data/data'

    # test_image_path = '/home/model/work/ai_data/test1'
    test_resize_path = '/home/model/work/ai_data/resize_test'


class TrainingArg:
    n_epochs = 200
    batch_size = 128
    update_rule = 'adam'
    learning_rate = 0.001
    lr_decay_steps = 3000
    print_bleu = True
    print_every = 1000
    save_every = 1
    log_path = './log/'
    model_path = './model/'
    pretrained_model = './model/model-386000'
    test_model = './model/model-386000'

    vgg19_path = './data_se/vgg19.mat'

    check_test = False
