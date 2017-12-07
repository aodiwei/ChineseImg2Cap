#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/10/26'
# 
"""
import os
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print(('Loaded %s..' % path))
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(('Saved %s..' % path))


def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' % (epoch + 1))
        f.write('Bleu_1: %f\n' % scores['Bleu_1'])
        f.write('Bleu_2: %f\n' % scores['Bleu_2'])
        f.write('Bleu_3: %f\n' % scores['Bleu_3'])
        f.write('Bleu_4: %f\n' % scores['Bleu_4'])
        #f.write('METEOR: %f\n' % scores['METEOR'])
        f.write('ROUGE_L: %f\n' % scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' % scores['CIDEr'])