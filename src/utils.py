#!/usr/bin/env python3
# coding=utf-8

import os
import time
import torch
import numpy as np
import logging
import hashlib
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, classification_report


class NNHistory:
    def __init__(self):
        """包含 loss, acc, & n_iter的dict，用于记录NN train/val 过程中的数据
        n_iter在计算loss/acc加权平均时使用，例如一个epoch中每次迭代的batch数不是定值时；默认为全1，即不考虑加权
        """
        self.data = {'loss': [], 'acc': [], 'n_iter': []}

    def append(self, loss, acc, n_iter=1):
        self.data['loss'].append(loss)
        self.data['acc'].append(acc)
        self.data['n_iter'].append(n_iter)

    def append_for_NER(self, loss, preds, labels):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        preds = preds.view(-1).detach().cpu().numpy()  # [bs*sql]
        labels = labels.view(-1).detach().cpu().numpy()

        n_tokens = torch.sum(labels >= 0).item()  # num of non-pad tokens
        n_correct = torch.sum(preds == labels).item()
        acc = n_correct / n_tokens

        self.append(loss, acc, n_tokens)

    def avg(self):
        loss = np.array(self.data['loss'])
        acc = np.array(self.data['acc'])
        n_iter = np.array(self.data['n_iter'])

        n_sum = float(np.sum(n_iter))

        loss_avg = np.sum(loss * n_iter) / n_sum  # dot product, 点乘
        acc_avg = np.sum(acc * n_iter) / n_sum

        return loss_avg, acc_avg

    def last(self):
        return self.data['loss'][-1], self.data['acc'][-1]


class NNFullHistory:
    def __init__(self):
        """储存所有pred & label记录，用于后续复杂计算，例如precision/recall/f1，confusion matrix"""
        self.predss = []  # list of list (of int): [[],[],...]
        self.labelss = []

        self.losses = []
        self.counts = []  # num of non-pad tokens

    def all_preds(self):
        _all_preds = []
        for preds in self.predss:
            _all_preds.extend(preds)
        return np.array(_all_preds)

    def all_labels(self):
        _all_labels = []
        for labels in self.labelss:
            _all_labels.extend(labels)
        return np.array(_all_labels)

    def append(self, loss, preds, labels, ignore_label=-1):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(preds, torch.Tensor):
            preds = preds.view(-1).detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.view(-1).detach().cpu().numpy()

        count = len(labels)
        if ignore_label is not None:
            mask = labels != ignore_label
            count = np.sum(mask)
            preds = preds[mask]
            labels = labels[mask]

        self.predss.append(preds)
        self.labelss.append(labels)

        self.losses.append(loss)
        self.counts.append(count)

    def avg_accuracy(self):
        all_preds = self.all_preds()
        all_labels = self.all_labels()
        assert all_preds.size == all_labels.size
        return np.sum(all_preds == all_labels) / all_preds.size

    def avg_loss(self):
        losses = np.array(self.losses)
        counts = np.array(self.counts)

        return (losses * counts).sum() / counts.sum()  # dot-product for weighted

    def avg_prf1_binary(self, neg_label):
        """preds和labels中int的种类为分类的类数，通常大于2；
        可以指定其中一种类别为negative，其他全部算positive，从而得到precision_recall_f1
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html"""
        all_preds = self.all_preds()
        all_labels = self.all_labels()

        pos_label = neg_label + 1
        all_preds[all_preds != neg_label] = pos_label
        all_labels[all_labels != neg_label] = pos_label

        p, r, f1, s = precision_recall_fscore_support(all_labels, all_preds, pos_label=pos_label, average='binary')
        return p, r, f1

    def avg_prf1_all(self, output_dict=True, label_tags=None):
        """ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html"""
        all_preds = self.all_preds()
        all_labels = self.all_labels()
        return classification_report(all_labels, all_preds, target_names=label_tags, output_dict=output_dict, digits=3)

    def avg_prf1_weight(self, remove_first=False):
        d = self.avg_prf1_all(output_dict=True)
        # Example of d:
        # {'0': {'precision': 0.695, 'recall': 0.753, 'f1-score': 0.723, 'support': 3728},
        # '1': {'precision': 0.875, 'recall': 0.327, 'f1-score': 0.476, 'support': 107},
        # '2': {'precision': 0.727, 'recall': 0.488, 'f1-score': 0.584, 'support': 481},
        # ...
        # '16': {'precision': 0.442, 'recall': 0.145, 'f1-score': 0.218, 'support': 586},
        # 'accuracy': 0.694,
        # 'macro avg': {'precision': 0.603, 'recall': 0.488, 'f1-score': 0.515, 'support': 17825},
        # 'weighted avg': {'precision': 0.696, 'recall': 0.694, 'f1-score': 0.684, 'support': 17825}}

        if remove_first:
            # del d['0']
            p, r, f1, n = 0, 0, 0, 0
            for i in range(1, 15):
                n1 = d[str(i)]['support']
                p += d[str(i)]['precision'] * n1
                r += d[str(i)]['recall'] * n1
                f1 += d[str(i)]['f1-score'] * n1
                n += n1
            return p / n, r / n, f1 / n
        else:
            dw = d['weighted avg']
            return dw['precision'], dw['recall'], dw['f1-score']


class _VerboseLogger:
    # print_level_threshold
    VeryVerbose = -1
    Verbose = 0
    Debug = 1
    Info = 2
    Important = 3
    VeryImportant = 4

    def __init__(self, print_level_threshold=Debug, file_dir='./logs/debug/'):
        """All messages will be saved to the log file.
        The lower print_level_threshold is, the more message will be printed
        print_level_threshold: 0 - 4: verbose, debug, info, warning, error"""

        self.print_level_threshold = print_level_threshold

        self.writer = open((file_dir + '{}.log').format(datetime.now().strftime('%m-%d')), 'a+')
        self.writer.write('\n[{}] {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''))

    def log(self, msg, level=Debug):
        self.writer.write('\n')
        self.writer.write(msg)
        if level >= self.print_level_threshold:
            print(msg)

    def log_print(self, msg):
        self.log(msg, level=self.print_level_threshold + 1)

    def log_only(self, msg):
        self.log(msg, level=self.print_level_threshold - 1)


class Logger:
    def __init__(self, file_dir='./logs/', file_name=None, init_mode='a+', print_log_level=1):
        if not file_name:
            file_name = f"log_{datetime.now().strftime('%m-%d')}.log"
        self.file_path = os.path.join(file_dir, file_name)
        self.print_log_level = print_log_level

        with open(self.file_path, init_mode) as f:
            if 'a' in init_mode:
                f.write('\n')
            f.write('[{}]\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def log(self, msg, end='\n', level=1, print_log=None, warning=False):
        """
        :param msg:
        :param end:
        :param level:       higher (importance) level, more likely to be print
        :param print_log:   None means auto, set True/False to control print
        :param warning:     use warning instead of print
        """
        if print_log is not None:
            if print_log:  # True
                level = self.print_log_level + 10
            else:  # False
                level = self.print_log_level - 10

        if level >= self.print_log_level:  # print
            if warning:
                # warnings.formatwarning = custom_formatwarning
                # warnings.warn(msg)
                logging.warning(msg)
            else:
                print(msg, end=end)

        self._write_log(f'{msg}{end}')  # log file

    def _write_log(self, msg):
        with open(self.file_path, 'a+') as f:
            f.write(msg)


def get_elapsed_time(start_time):
    dt = time.time() - start_time
    if dt < 1:
        str_ = '{:.4f} s'.format(dt)
    elif dt < 60:
        str_ = '{:.2f} s'.format(dt)
    else:
        str_ = '{:.1f} min'.format(dt / 60.0)
    return str_


def str_hash(x, length=7):
    if not isinstance(x, str):
        x = str(x)

    hash_str = hashlib.sha1(x.encode('utf8')).hexdigest()
    if length is None or length <= 0:
        return hash_str
    else:
        return hash_str[:length]
