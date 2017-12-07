#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/10/26'
# 
"""
import os
from collections import Counter
import json
import random

import jieba
from PIL import Image
from scipy import ndimage
import numpy as np

from core.vggnet import Vgg19
from tool import utils
from config.config_const import TrainingArg


class PreData:
    def __init__(self, vgg19_path=None):
        if vgg19_path is not None:
            self.vggnet = Vgg19(vgg19_path)
            self.vggnet.build()

        self.sess = None
        self.val_data = None

    def set_tf_sess(self, sess):
        """
        set sess
        :param sess:
        :return:
        """
        self.sess = sess

    def build_vocab(self, caption_train_path, max_words_len_path, vocab_path, threshold=1):
        with open(caption_train_path, 'r') as f:
            caption_json_list = json.load(f)
        counter = Counter()
        max_len = 0
        for i, item in enumerate(caption_json_list):
            for cap in item['caption']:
                print('\nsentence: {}'.format(cap))
                words = jieba.cut(cap, cut_all=False)
                words_len = 0
                for w in words:
                    words_len += 1
                    counter[w] += 1
                    # print(w, end=' ')
                max_len = words_len if words_len > max_len else max_len
                # break

        vocab = [word for word, v in counter.most_common() if v >= threshold]  # sort
        print(('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold)))

        word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
        idx = 3
        for word in vocab:
            word_to_idx[word] = idx
            idx += 1
        print("Max length of caption: ", max_len)
        with open(max_words_len_path, 'w') as f:
            f.write(str(max_len))
        utils.save_pickle(word_to_idx, vocab_path)

    def load_vocab(self, vocab_path):
        """
        
        :return: 
        """
        return utils.load_pickle(vocab_path)

    def build_caption_vector(self, max_words_len_path, input_path, output_path, vocab_path, image_root_path=None):
        """
        transfer
        :return: 
        """
        with open(max_words_len_path, 'r') as f:
            max_length = int(f.read())
            print('max_words_len: {}'.format(max_length))

        word_to_idx = self.load_vocab(vocab_path)
        with open(input_path, 'r') as f:
            caption_json_list = json.load(f)
        int_vector = []
        for i, item in enumerate(caption_json_list):
            if image_root_path:
                p = os.path.join(image_root_path, item['image_id'])
                if not os.path.exists(p):
                    print('invalid image {}'.format(p))
                    continue
            new_item = {'image_id': item['image_id']}
            caption = []
            for cap in item['caption']:
                caption.append(self._caption_vector(cap, word_to_idx, max_length))
            new_item['caption'] = caption
            int_vector.append(new_item)

            # break

        utils.save_pickle(int_vector, output_path)

    def _caption_vector(self, caption, word_to_idx, max_length):
        """
        
        :param word_to_idx: 
        :return: 
        """

        words = jieba.cut(caption, cut_all=False)
        cap_vec = [word_to_idx['<START>']]
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        len_cap_vec = len(cap_vec)
        if len_cap_vec < (max_length + 2):
            extend_len = max_length + 2 - len_cap_vec
            cap_vec.extend([word_to_idx['<NULL>']] * extend_len)

        return cap_vec

    def resize_image(self, image):
        width, height = image.size
        if width > height:
            left = (width - height) / 2
            right = width - left
            top = 0
            bottom = height
        else:
            top = (height - width) / 2
            bottom = height - top
            left = 0
            right = width
        image = image.crop((left, top, right, bottom))
        image = image.resize([224, 224], Image.ANTIALIAS)
        return image

    def resize_images(self, in_image_path, out_path):
        """
        
        :param out_path: 
        :param in_image_path: 
        :return: 
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print('Start resizing %s images.' % in_image_path)
        image_files = os.listdir(in_image_path)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(in_image_path, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = self.resize_image(image)
                    image.save(os.path.join(out_path, image_file), image.format)
            if i % 100 == 0:
                print('Resized images: %d/%d' % (i, num_images))

    def decode_captions(self, captions, idx_to_word, ignore_start=False):
        """

        :param captions:
        :param idx_to_word:
        :return:
        """
        if captions.ndim == 1:
            T = captions.shape[0]
            N = 1
        else:
            N, T = captions.shape

        decoded = []
        for i in range(N):
            words = []
            for t in range(T):
                if captions.ndim == 1:
                    word = idx_to_word[captions[t]]
                else:
                    word = idx_to_word[captions[i, t]]
                if word == '<START>' and ignore_start:
                    continue
                if word == '<END>':
                    words.append('.')
                    break
                if word != '<NULL>':
                    words.append(word)
            decoded.append(' '.join(words))
        return decoded

    def pre_val_to_train(self, train_vect_in_path, val_in_path):
        """

        :param train_vect_in_path:
        :param val_in_path:
        :return:
        """
        train_vect = utils.load_pickle(train_vect_in_path)
        val_vect = utils.load_pickle(val_in_path)
        n = 1280
        train_vect = train_vect + val_vect[n:]
        val_vect = val_vect[:n]

        utils.save_pickle(train_vect, train_vect_in_path)
        utils.save_pickle(val_vect, val_in_path)

    def pre_data_with_vgg19(self, image_dir, filenames):
        """
        pre train
        :param filenames: list
        :param image_dir: 
        :return:
        """
        if self.sess is None:
            raise ValueError('set sess first')
        if TrainingArg.check_test:
            image_batch_file = ['./data/test.jpg'] * 128
            print('WARNING: CHECK TEST MODEL NOW')
        else:
            image_batch_file = [os.path.join(image_dir, x) for x in filenames]
        image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(np.float32)
        return self.sess.run(self.vggnet.features, feed_dict={self.vggnet.images: image_batch})

    def fetch_batch(self, caption_data_path, image_data_path, batch_size, epochs):
        """

        :param caption_data_path:
        :param image_data_path:
        :param batch_size:
        :param epochs:
        :return:
        """
        captions = utils.load_pickle(caption_data_path)
        self.train_data_count = len(captions)

        for epoch in range(4, epochs):
            start = 0
            end = batch_size
            random.shuffle(captions)
            for item in range(0, self.train_data_count, batch_size):
                temp = captions[start:end]
                random.shuffle(temp)
                if epoch < 5:
                    cap_index = epoch % 5
                else:
                    cap_index = random.randint(0, 4)
                caption_batch = []
                image_batch = []
                for x in temp:
                    caption_batch.append(x['caption'][cap_index])
                    image_batch.append(x['image_id'])
                try:
                    image_batch = self.pre_data_with_vgg19(image_data_path, image_batch)
                except Exception as e:
                    print('load image failed: {}-{}, cause {}'.format(start, end, e))
                    start += batch_size
                    end += batch_size
                    continue
                start += batch_size
                end += batch_size
                yield caption_batch, image_batch, epoch

    def fetch_val_batch(self, caption_data_path, image_data_path, batch_size):
        """

        :param caption_data_path:
        :param image_data_path:
        :param batch_size:
        :return:
        """
        captions = utils.load_pickle(caption_data_path)
        val_data_count = len(captions)
        if TrainingArg.check_test:
            val_data_count = batch_size
            print('WARNING: CHECK TEST MODEL NOW')
        start = 0
        end = batch_size
        for item in range(0, val_data_count, batch_size):
            temp = captions[start:end]
            caption_batch = []
            image_batch = []
            for x in temp:
                caption_batch.append(x['caption'])
                image_batch.append(x['image_id'])
            try:
                image_batch = self.pre_data_with_vgg19(image_data_path, image_batch)
            except Exception as e:
                print('load image failed: {}-{}, cause {}'.format(start, end, e))
                start += batch_size
                end += batch_size
                continue
            start += batch_size
            end += batch_size
            yield caption_batch, image_batch

    def pre_orig_image_to_tell(self, filename):
        """
        process original
        :param filename: 
        :return: 
        """
        with open(filename, 'r+b') as f:
            with Image.open(f) as image:
                image = self.resize_image(image)
                resize_path = filename.replace('.jpg', '_resize.jpg')
                image.save(resize_path, image.format)

        if self.sess is None:
            raise ValueError('set sess first')
        image_batch_file = [resize_path] * 2
        image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(np.float32)
        feature = self.sess.run(self.vggnet.features, feed_dict={self.vggnet.images: image_batch})

        return feature, resize_path

    def fetch_test_data(self, path, batch_size=128):
        """
        
        :param batch_size: 
        :param path: 
        :return: 
        """
        files = os.listdir(path)
        test_data_count = len(files)
        start = 0
        end = batch_size
        for item in range(0, test_data_count, batch_size):
            temp = files[start:end]
            try:
                features = self.pre_data_with_vgg19(path, temp)
                image_id = [x.rstrip('.jpg') for x in temp]
            except Exception as e:
                print('load image failed: {}-{}, cause {}'.format(start, end, e))
                start += batch_size
                end += batch_size
                continue
            start += batch_size
            end += batch_size
            yield features, image_id
