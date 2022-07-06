#!/usr/bin/env python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertZhTokenClassifier_(nn.Module):
    def __init__(self, n_labels, p_drop=0.1, bert_name='./models/bert-base-chinese'):
        super().__init__()
        self.n_labels = n_labels

        self.bert = BertModel.from_pretrained(bert_name, local_files_only=True)
        self.dropout = nn.Dropout(p_drop)
        in_features = 1024 if 'large' in bert_name else 768
        self.classifier = nn.Linear(in_features, self.n_labels)  # bert-base's hidden size=768/1024

    def forward(self, inputs, att_mask):
        x, _ = self.bert(inputs, attention_mask=att_mask)
        x = self.dropout(x)

        x = self.classifier(x)
        return x
