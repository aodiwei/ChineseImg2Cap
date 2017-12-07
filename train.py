#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/10/26'
#
"""
import os
import time

import datetime
import tensorflow as tf
import numpy as np
# from uaitrain.arch.tensorflow import uflag

from core.model import CaptionGenerator
from core.bleu import evaluate
from config.config_const import Const, TrainingArg
from tool import utils
from preproccess_data.pre_data import PreData

FLAGS = tf.app.flags


class Trainer:
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
        self.log_path = TrainingArg.log_path  # FLAGS.log_dir
        self.model_path = TrainingArg.model_path  # FLAGS.output_dir  # TrainingArg.model_path
        self.data_dir = Const.resize_train_out_path  # FLAGS.data_dir
        self.pretrained_model = TrainingArg.pretrained_model
        self.test_model = TrainingArg.test_model
        self.max_words_len = 35

        self.pre_mgr = PreData(vgg19_path=TrainingArg.vgg19_path)  # 数据管理

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.org_decoded = {}
        self.val_data_flag = False

    def train(self):
        """
        training
        :return:
        """
        loss = self.model.build_model()

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            with tf.name_scope('optimizer'):
                tf.get_variable_scope().reuse_variables()
                _, _, generated_captions = self.model.build_sampler(max_len=self.max_words_len)

                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                lr = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step,
                                                decay_steps=TrainingArg.lr_decay_steps,
                                                decay_rate=0.96, staircase=True, name='learn_rate')
                optimizer = self.optimizer(learning_rate=lr)
                grads = tf.gradients(loss, tf.trainable_variables())
                grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.global_step)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradient', grad)

        summary_op = tf.summary.merge_all()

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.pre_mgr.set_tf_sess(sess)

            tf.initialize_all_variables().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=10)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            curr_epoch = 0
            batchs = self.pre_mgr.fetch_batch(Const.caption_train_vector_path, self.data_dir,
                                              self.batch_size, self.n_epochs)

            for batch in batchs:
                caption_batch, image_batch, epoch = batch
                feed_dict = {self.model.features: image_batch, self.model.captions: caption_batch}
                _, l, step = sess.run([train_op, loss, self.global_step], feed_dict)

                if step % self.print_every == 0 or step == 1:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, step)
                    print("\nTrain loss at epoch %d & step %d (mini-batch): %.5f" % (epoch + 1, step, l))
                    # ground_truths = captions[image_idxs == image_idxs_batch[0]]
                    ground_truths = np.array([caption_batch[0]])
                    decoded = self.pre_mgr.decode_captions(ground_truths, self.model.idx_to_word)
                    for j, gt in enumerate(decoded):
                        print("Ground truth %d: %s" % (j + 1, gt))
                    gen_caps = sess.run(generated_captions, feed_dict)
                    decoded = self.pre_mgr.decode_captions(gen_caps, self.model.idx_to_word)
                    print("Generated caption: %s\n" % decoded[0])

                print('{}, epoch:{} step: {}，Current epoch loss: {}'.format(datetime.datetime.now().isoformat(), epoch + 1, step, l))

                # print(out BLEU scores and file write
                if curr_epoch != epoch or step == 1 or step % self.print_every == 0:
                    curr_epoch = epoch
                    val_data_batchs = self.pre_mgr.fetch_val_batch(Const.val_vector_out_path, self.data_dir, self.batch_size)
                    gen_caps = []
                    i = 0
                    for val_batch in val_data_batchs:
                        val_caption, val_image = val_batch
                        # features_batch = val_features[i * self.batch_size:(i + 1) * self.batch_size]
                        feed_dict = {self.model.features: val_image}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        gen_caps.extend(gen_cap)
                        if not self.val_data_flag:
                            print('val batch loop {}'.format(i))
                            for item in val_caption:
                                self.org_decoded[i] = self.pre_mgr.decode_captions(np.array(item), self.model.idx_to_word,
                                                                                   ignore_start=True)
                                i += 1
                                # break
                    self.val_data_flag = True
                    gen_decoded = self.pre_mgr.decode_captions(np.array(gen_caps), self.model.idx_to_word)
                    for j in range(5):
                        print('val org sents: {}'.format(self.org_decoded[j]))
                        print('val gen sents: {}\n'.format(gen_decoded[j]))

                    scores = evaluate(gen_decoded, self.org_decoded, get_scores=True)
                    utils.write_bleu(scores=scores, path=self.model_path, epoch=epoch)

                    # save model's parameters
                    # if (e + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=step)
                    print("model-%s saved." % (epoch + 1))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
