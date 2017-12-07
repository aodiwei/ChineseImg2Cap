#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/10/26'
# 
"""
import os

from config.config_const import Const
from preproccess_data.pre_data import PreData
from train import Trainer


def pre_data():
    """
    
    :return: 
    """
    pre_d = PreData()
    pre_d.build_vocab(caption_train_path=Const.caption_train_path, vocab_path=Const.vocab_path,
                      max_words_len_path=Const.max_words_len_path)


def pre_build_caption_vector():
    """
    
    :return: 
    """
    pre_d = PreData()
    pre_d.build_caption_vector(max_words_len_path=Const.max_words_len_path, input_path=Const.caption_train_path,
                               output_path=Const.caption_train_vector_path, vocab_path=Const.vocab_path,
                               image_root_path=Const.resize_train_out_path)

    pre_d.build_caption_vector(max_words_len_path=Const.max_words_len_path, input_path=Const.val_caption_path,
                               output_path=Const.val_vector_out_path, vocab_path=Const.vocab_path,
                               image_root_path=Const.val_resize_path)


def pre_resize_to_vgg19_train_images():
    """
    
    :return: 
    """
    pre_d = PreData()
    # pre_d.resize_images(Const.train_image_path, Const.resize_train_out_path)
    pre_d.resize_images(Const.val_image_path, Const.val_resize_path)
    pre_d.resize_images(Const.test_image_path, Const.test_resize_path)


def run_train():
    """

    :return:
    """
    trainer = Trainer()
    trainer.train()


def pre_merge():
    """

    :return:
    """
    pre_d = PreData()
    pre_d.pre_val_to_train(Const.caption_train_vector_path, Const.val_vector_out_path)


if __name__ == '__main__':
    # pre_data()
    # pre_build_caption_vector()
    # pre_resize_to_vgg19_train_images()
    run_train()
    # pre_merge()
