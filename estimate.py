#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/11/1'
# 
"""
import json

import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import tensorflow as tf
import time
import os
import pickle
from scipy import ndimage

from config.config_const import Const, TrainingArg
from core.model import CaptionGenerator
from preproccess_data.pre_data import PreData
from tool import utils

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class Estimate:
    def __init__(self):
        self.word_to_idx = utils.load_pickle(Const.vocab_path)
        self.model = CaptionGenerator(self.word_to_idx,
                                      dim_feature=[196, 512],
                                      dim_embed=512,
                                      dim_hidden=1024,
                                      n_time_step=33,
                                      prev2out=True,
                                      ctx2out=True,
                                      alpha_c=1.0,
                                      selector=True,
                                      dropout=True)

        self.n_epochs = TrainingArg.n_epochs
        self.batch_size = TrainingArg.batch_size
        self.update_rule = TrainingArg.update_rule
        self.learning_rate = TrainingArg.learning_rate
        self.print_bleu = TrainingArg.print_bleu
        self.print_every = TrainingArg.print_every
        self.save_every = TrainingArg.save_every
        self.log_path = TrainingArg.log_path
        self.model_path = TrainingArg.model_path
        self.pretrained_model = TrainingArg.pretrained_model
        self.test_model = TrainingArg.test_model
        self.max_words_len = 35

        self.pre_mgr = PreData(vgg19_path=TrainingArg.vgg19_path)  # 数据管理

    def test(self, image_path):
        """
        
        :return: 
        """
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.max_words_len)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, TrainingArg.test_model)
            # saver.restore(sess, r'F:\4_study\show_tell\show-attend-and-tell-master\model\lstm\model-1')
            self.pre_mgr.set_tf_sess(sess)

            feature, resize_path = self.pre_mgr.pre_orig_image_to_tell(image_path)
            feed_dict = {self.model.features: feature}

            # batchs = self.pre_mgr.fetch_batch(Const.caption_train_vector_path, Const.resize_train_out_path,
            #                                   self.batch_size, self.n_epochs)
            #
            # for batch in batchs:
            #     caption_batch, image_batch, epoch = batch
            #     feed_dict = {self.model.features: image_batch}

            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            #sam_cap = sess.run(sampled_captions, feed_dict)
            decoded = self.pre_mgr.decode_captions(sam_cap, self.model.idx_to_word)
            print(decoded)
            n = 0
            # Plot original image
            # resize_path = ''
            img = ndimage.imread(resize_path)
            plt.subplot(4, 5, 1)
            plt.imshow(img)
            plt.axis('off')

            # Plot images with attention weights
            words = decoded[n].split(" ")
            for t in range(len(words)):
                if t > 18:
                    break
                plt.subplot(4, 5, t + 2)
                plt.text(0, 1, '%s(%.2f)' % (words[t], bts[n, t]), color='black', backgroundcolor='white', fontsize=8)
                plt.imshow(img)
                alp_curr = alps[n, t, :].reshape(14, 14)
                alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                plt.imshow(alp_img, alpha=0.85)
                plt.axis('off')
            plt.show()

    def test_data(self, path):
        """
        
        :return: 
        """
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.max_words_len)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, TrainingArg.test_model)
            self.pre_mgr.set_tf_sess(sess)

            test_result = []
            test_batch = self.pre_mgr.fetch_test_data(path)
            for feature, image_id in test_batch:
                feed_dict = {self.model.features: feature}
                sam_cap = sess.run([sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
                decoded = self.pre_mgr.decode_captions(sam_cap, self.model.idx_to_word)
                for i, v in enumerate(decoded):
                    item = {
                        'image_id': image_id[i],
                        'caption': decoded[i].replace(' ', '').rstrip('.')
                    }
                    test_result.append(item)
                print(test_result[-1])

            with open('adw_image_caption.json', 'w') as f:
                json.dump(test_result, f)
                print('save test json to adw_image_caption.json')


if __name__ == '__main__':
    path = r'F:\4_study\ai_data\test_temp\caption-eg1.jpg'
    path = r'F:\4_study\ai_data\pre_Data\test_temp'
    est = Estimate()
    # est.test(path)
    path = Const.test_resize_path
    est.test_data(path)


