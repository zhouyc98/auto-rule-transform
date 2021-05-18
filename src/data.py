#!/usr/bin/python3.7
# coding=utf-8

import os
import sys
import json
import re
import pickle
import torch
import platform
import numpy as np
from torch.utils import data
from sklearn.utils import shuffle
# from sklearn.model_selection import StratifiedKFold
from utils import *


class Corpus:
    def __init__(self, data_dir, max_sql=125, is_test=False):
        from transformers import BertTokenizer

        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
        self.max_sql = max_sql

        self.tags = self.get_tags()  # 若要对属性更名，记得删除pickle的dat文件，重新生成一次数据
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tags)}

        if is_test:
            self.test_dataset = TextDataSet(self.load_test_data())
        else:
            self.val_dataset = TextDataSet(self.load_data('val'))
            self.train_dataset = TextDataSet(self.load_data('train'))

    def ids_to_tags(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        tags = []
        for id in ids:
            tags.append(self.idx2tag[id])
        return tags

    def tags_to_ids(self, tags):
        ids = []
        for tag in tags:
            ids.append(self.tag2idx[tag])
        return ids

    def get_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                if tag.strip():
                    tags.append(tag.strip())
        return tags

    def load_data(self, data_type='train'):
        """
        :param data_type: 'train'/'val'
        :return:
            sen_token_ids: list of sentences, represented by token id. [[101,102,...],[200,300,...],...]
            sen_att_mask:
            sen_tag_ids: list of sentences' tags, represented by tag id. [[2,1,5,...],[3,1,2,...],...]
        """
        with open(os.path.join(self.data_dir, data_type, 'sentences.txt'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            # encode_plus = tokenize + to_ids + to_tensor
            # bert-chinese中，中文之间的空格会被去掉
            _sen_tokens = self.tokenizer.batch_encode_plus(lines, max_length=self.max_sql, pad_to_max_length=True)
            # # ===== Stats
            # import matplotlib.pyplot as plt
            # att_masks = _sen_tokens['attention_mask']
            # lens = np.array(list(map(lambda x: sum(x), att_masks)))
            # plt.hist(lens, bins=50)
            # plt.show()
            # # sql: 80=0.959, 90=0.975, 100=0.975, 125=1.0
            # for n in (80, 85, 90, 95, 100, 110, 125, 150):
            #     print(f'{n}:', np.sum(lens < n) / len(lens))
            # breakpoint()

        sen_token_ids = torch.tensor(_sen_tokens['input_ids'], dtype=torch.long)
        sen_att_mask = torch.tensor(_sen_tokens['attention_mask'], dtype=torch.float)
        # self.tokenizer.convert_ids_to_tokens(sen_token_ids[0])
        CLS_token_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]  # 101

        sen_tag_ids = -1 * torch.ones_like(sen_token_ids, dtype=torch.long)  # -1 for tag padding (non-labeled)
        with open(os.path.join(self.data_dir, data_type, 'tags.txt'), 'r') as file:
            for i, line in enumerate(file):
                ### token_ids[0] == [CLS], shift toward right
                assert sen_token_ids[i][0].item() == CLS_token_id
                tag_ids = [self.tag2idx[tag] for tag in line.strip().split(' ')]
                if len(tag_ids) > self.max_sql - 1:
                    tag_ids = tag_ids[:self.max_sql - 1]
                sen_tag_ids[i, 1:1 + len(tag_ids)] = torch.tensor(tag_ids, dtype=torch.long)

        d = {'token_ids': sen_token_ids, 'att_mask': sen_att_mask, 'tag_ids': sen_tag_ids}
        return d

    def load_test_data(self):
        """ test data: no label/tag
        :return:
            sen_token_ids: list of sentences, represented by token id. [[101,102,...],[200,300,...],...]
            sen_att_mask:
        """
        test_data_dir = os.path.join(self.data_dir, 'test')

        lines = []
        fns = os.listdir(test_data_dir)
        for fn in fns:
            if '.txt' not in fn:
                continue
            print(f'Read {fn}')
            ffn = os.path.join(test_data_dir, fn)  # full file name
            with open(ffn, 'r', encoding='utf8') as fp:
                lines1 = fp.readlines()
                lines.extend(lines1)

        ### Clean
        def clean_seq(seq_):
            # 第一个空格之前，如果全是ASCII（如‘1.2.3 ’），则删掉
            if ' ' in seq_:
                i = seq_.index(' ')
                if all(ord(c) < 128 for c in seq_[:i]):
                    seq_ = seq_[i + 1:]

            seq_ = seq_.replace(' ', '')
            return seq_

        for i in range(len(lines)):
            lines[i] = clean_seq(lines[i])
        ###

        _sen_tokens = self.tokenizer.batch_encode_plus(lines, max_length=self.max_sql, pad_to_max_length=True)
        sen_token_ids = torch.tensor(_sen_tokens['input_ids'], dtype=torch.long)
        sen_att_mask = torch.tensor(_sen_tokens['attention_mask'], dtype=torch.float)
        # CLS_token_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]  # 101

        d = {'token_ids': sen_token_ids, 'att_mask': sen_att_mask}
        return d

    def check_tags_stratify(self, print_result=False):
        import pandas as pd

        t = self.train_dataset.sen_tag_ids.numpy().reshape(-1)
        v = self.val_dataset.sen_tag_ids.numpy().reshape(-1)
        tv = np.concatenate((t, v))
        tags = {'tag-id': [], 'tag': [], 'count-all': [], 'count-train': [], 'count-val': [], 'train-val-ratio': []}
        for i in range(15):
            tags['tag-id'].append(i)
            tags['tag'].append(self.idx2tag[i])
            tags['count-all'].append(np.sum(tv == i))
            tags['count-train'].append(np.sum(t == i))
            tags['count-val'].append(np.sum(v == i))
            tags['train-val-ratio'].append(round(tags['count-train'][-1] / tags['count-val'][-1], 1))

        tags_df = pd.DataFrame(tags)
        if print_result:
            print(tags_df)
            print(f"Min, Max of train-val-ratio: {min(tags_df['train-val-ratio'])}, {max(tags_df['train-val-ratio'])}")

        check_pass = True
        if 0 in tags_df['count-val'].to_numpy() or 0 in tags_df['count-train'].to_numpy():
            check_pass = False

        return check_pass, tags_df

    def get_token_ids_bool_mask(self, token_ids: torch.Tensor):
        cls_id, pad_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[PAD]', '[SEP]'])  # 101,0,102
        mask = ((token_ids != pad_id).float() * (token_ids != cls_id).float() * (token_ids != sep_id).float()).bool()
        return mask

    def render_seq_labels(self, seq, label, pred, mark_down=False):
        # if (label is None) and (pred is None):
        #     return ''.join(seq)

        if isinstance(seq, torch.Tensor):
            # seq = seq.view(-1)
            mask = label >= 0 if label is not None else self.get_token_ids_bool_mask(seq)

            seq = self.tokenizer.convert_ids_to_tokens(seq.masked_select(mask))
            if label is not None:
                label = self.ids_to_tags(label.masked_select(mask))
            if pred is not None:
                pred = self.ids_to_tags(pred.masked_select(mask))

        if label is not None:
            label_iit = label_bio_to_iit(label, seq)
            # print(self._render_seq_label_lines(seq, label_iit))
            label_str = self._render_seq_label(seq, label_iit, mark_down)
            # print(label_str)
        else:
            label_str = None

        if pred is not None:
            pred_iit = label_bio_to_iit(pred, seq)
            pred_str = self._render_seq_label(seq, pred_iit, mark_down)
        else:
            pred_str = None

        seq_str = ''.join(seq).replace('[', '<').replace(']', '>')
        return label_str, pred_str, seq_str

    def _render_seq_label(self, seq, label, mark_down=False):
        """
        :param seq: list of char
        :param label: label_iit
        :return: e.g., [水带/obj]，在设计工作压力下其[轴向延伸率/prop]
        """
        seq = seq.copy()
        for i in range(len(seq)):
            seq[i] = seq[i].replace('[', '<').replace(']', '>')

        for i, j, tag in label:
            s = seq[i] if j <= i + 1 else ''.join(seq[i:j])
            if mark_down:
                seq[i] = f' `{s}/{tag}` '
            else:
                seq[i] = f'[{s}/{tag}]'

            if j > i + 1:
                for k in range(i + 1, j):
                    seq[k] = ''

        return (''.join(seq)).strip()

    def _render_seq_label_lines(self, seq, label):
        seq = seq.copy()
        for i in range(len(seq)):
            seq[i] = seq[i].replace('[', '<').replace(']', '>')

        seq_lb = []
        for i, j, tag in label:
            seq_lb.append(''.join(seq[i:j]) + ' ' + tag)

        return '\n'.join(seq_lb)


class TextDataSet(data.Dataset):
    def __init__(self, data: dict):
        self.sen_token_ids = data['token_ids']
        self.sen_att_mask = data['att_mask']
        if 'tag_ids' in data:  # consider test data
            self.sen_tag_ids = data['tag_ids']

        self.len = len(self.sen_token_ids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if hasattr(self, 'sen_tag_ids'):
            return self.sen_token_ids[index], self.sen_att_mask[index], self.sen_tag_ids[index]

        return self.sen_token_ids[index], self.sen_att_mask[index]


def get_data_loader(data_dir, batch_size, max_sql=125, enable_save=False, check_stratify=True, shuffle_train=True):
    """
    example:
    for token_ids, att_mask, tag_ids in val_data_loader:
        print(token_ids.shape, att_mask.shape, tag_ids.shape)
        # torch.Size([32, 125]) torch.Size([32, 125]) torch.Size([32, 125])
    """

    if enable_save:
        corpus_path = os.path.join(data_dir, 'corpus.dat')
        if not os.path.exists(corpus_path):
            print('Processing...')
            corpus = Corpus(data_dir, max_sql)
            with open(corpus_path, 'wb') as fp:
                pickle.dump(corpus, fp)
        else:
            with open(corpus_path, 'rb') as fp:
                corpus = pickle.load(fp)
    else:
        corpus = Corpus(data_dir, max_sql)

    if check_stratify:
        check, tags_df = corpus.check_tags_stratify()
        assert check, f'Check tags stratify failed. try another random_sate in shuffle for data\n{tags_df}'
        # print(tags_df)

    num_workers = 0 if 'Windows' in platform.platform() else 4
    train_data_loader = data.DataLoader(dataset=corpus.train_dataset, batch_size=batch_size,
                                        shuffle=shuffle_train, num_workers=num_workers, drop_last=False)
    val_data_loader = data.DataLoader(dataset=corpus.val_dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers, drop_last=False)

    return train_data_loader, val_data_loader, corpus


def get_test_data_loader(data_dir, batch_size, max_sql=125):
    """
    example:
    for token_ids, att_mask in test_data_loader:
        print(token_ids.shape, att_mask.shape)
        # torch.Size([32, 125]) torch.Size([32, 125]))
    """
    corpus = Corpus(data_dir, max_sql, is_test=True)

    num_workers = 0 if 'Windows' in platform.platform() else 4
    test_data_loader = data.DataLoader(dataset=corpus.test_dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers, drop_last=False)

    return test_data_loader, corpus


# ========================================================================================= Process
def process_xiaofang_data(data_dir=r'C:\Users\Zhou Yucheng\硕士\Researches\自动合规性审查\Data\xiaofang'):
    docs = []
    doc_sep = '\n\n#=====\n\n'

    folders = os.listdir(data_dir)
    for fd in folders:
        if fd not in ['0', '1', '2', '3']:
            continue
        fd = os.path.join(data_dir, fd)
        json_paths = os.listdir(fd)
        for json_path in json_paths:
            with open(os.path.join(fd, json_path), 'r', encoding='utf8') as f:
                doc_js = json.load(f)
            docs.append(json_to_doc(doc_js))

    docs = np.array(docs)
    np.random.shuffle(docs)

    # n = docs.shape[0]
    # n_train, n_val = int(n * 0.8), int(n * 0.1)
    # docs_dict = {'train': docs[:n_train], 'val': docs[n_train:n_train + n_val], 'test': docs[n_train + n_val:]}
    # for t, docs_1 in docs_dict.items():
    with open(os.path.join(data_dir, 'processed/xiaofang.txt'), 'w', encoding='utf8') as f:
        for doc in docs:
            f.write(doc_sep)
            f.write(doc)


def json_to_doc(doc_js):
    """ keys: {'name', 'about', 'content'} & //{'code'}
    keys in ['content']: {'explain', 'title', 'body'} & //{'strong', 'hasChildren', 'chapter'}
    """

    def process_txt(txt: str):
        """
        1) XXX\n\u003csup\u003e2\u003c/sup\u003e -----> XXX\n<sup>2</sup> -----> XXX2
        2) remove <img> tag
        3) clean redundant \n
        """
        if txt.strip() == '':
            return ''

        # txt = eval("'" + di + "'") # unnecessary
        if '<img>' in txt:
            txt = txt.replace('<img>', '')

        if '<' in txt:
            txt = re.sub(r'\n<(sub|sup)>(.*)</(sub|sup)>', r'\2', txt)
            txt = re.sub(r'<(sub|sup)>(.*)</(sub|sup)>', r'\2', txt)
        txt = txt.replace('m\n2', 'm2')

        if txt.endswith('\n'):
            txt = txt.strip() + '\n'
        txt = txt.replace('\n\n\n', '\n\n')

        return txt

    doc_s = [doc_js['name'], '\n']  # use list instead of str concat, for performance
    if 'about' in doc_js:
        doc_s.extend([doc_js['about'], '\n'])

    if 'content' in doc_js and isinstance(doc_js['content'], list) and len(doc_js['content']) > 0:
        for i in range(len(doc_js['content'])):
            di = doc_js['content'][i]
            for k in ['explain', 'title', 'body']:
                if k in di:
                    doc_s.append('\n')
                    doc_s.append(process_txt(di[k]))

    doc = ''.join(doc_s)
    doc = doc.replace('\n\n\n\n\n\n', '\n\n').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

    return doc


def select_xiaofang_doc(data_dir=r'C:\Users\Zhou Yucheng\硕士\Researches\自动合规性审查\Data\xiaofang\processed'):
    doc = open(os.path.join(data_dir, 'xiaofang.txt'), 'r', encoding='utf8').read()
    doc = doc.replace('\n#=====\n', '。\n').replace('\n\n', '。\n').replace('\n', ' ')
    sents = re.split('[。？！]', doc)
    sents = [s.strip() + '\n' for s in sents if s.strip()]
    with open(os.path.join(data_dir, 'sentences.txt'), 'w', encoding='utf8') as f:
        f.writelines(sents)

    cmp_rel = '等于 大于 小于 大于等于 小于等于 大于或等于 小于或等于 不大于 不小于 高于 低于 不超过'.split()
    for i in range(len(sents) - 1, -1, -1):
        if '；' in sents[i]:  # split ';'
            sents[i] = sents[i].split('；')[0]
        if not any([cr in sents[i] for cr in cmp_rel]):
            del sents[i]
    for i in range(len(sents) - 1, -1, -1):
        if sents[i] in sents[:i]:  # 去重
            del sents[i]

    with open(os.path.join(data_dir, 'sentences_sel/sentences_sel_all.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sents))

    n_file = 11
    n_line = int(len(sents) / n_file)
    for i in range(n_file):
        with open(os.path.join(data_dir, 'sentences_sel/sentences_sel_{}.txt'.format(i)), 'w', encoding='utf8') as f:
            f.write('\n'.join(sents[i * n_line:(i + 1) * n_line]))


def init_data_by_json(data_dir='../data/xiaofang/', early_return=False, random_state=29):
    """ 根据Doccano生成的标注json文件，在train/val两个文件夹下生成sentences.txt, tags.txt
        early_return: 不修改train/ val/文件夹内容，read & update json文件后直接return """

    def _write_lines(path_, lines, add_space=False, add_enter=True):
        with open(path_, 'w', encoding='utf8') as f:
            for line in lines:
                f.write(' '.join(line) if add_space else line)
                if add_enter:
                    f.write('\n')

    def _json_pretty_dump(dicts, path):
        with open(path, 'w', encoding='utf8') as fp:
            # json.dump(dicts, fp, ensure_ascii=False)
            jstr = json.dumps(dicts, ensure_ascii=False)
            replaces = [('}, ', '\n},\n'), ('[{', '[\n{'), ('}]', '\n}\n]'), ('", "', '",\n"'), ('], "', '],\n"'),
                        ('{"', '{\n"'), ('"text', '    "text'), ('"label', '    "label'), ('"slabel', '    "slabel')]
            for r1, r2 in replaces:
                jstr = jstr.replace(r1, r2)
            fp.write(jstr)

    # ============================================================ Read & parse json
    pathjoin = os.path.join

    # # ========== jsonl merge to one
    # seqs, labels = [], []
    # for fn in os.listdir(pathjoin(data_dir, 'bak-jsons')):
    #     if '.json' in fn and 'all.json' not in fn:
    #         print(f'Read {fn} ...')
    #         ffn = pathjoin(data_dir, 'bak-jsons', fn)  # full file name
    #         with open(ffn, 'r', encoding='utf8') as fp:
    #             lines = fp.readlines()
    #         dicts = [json.loads(l) for l in lines]
    #         dicts = [d for d in dicts if d['labels']]  # filter out non-labeled sentences
    #         seqs1, labels1 = list(zip(*[(d['text'], d['labels']) for d in dicts]))
    #         seqs.extend(seqs1)
    #         labels.extend(labels1)
    # dicts = []
    # for seq, label in zip(seqs, labels):
    #     label.sort(key=lambda t: t[0])
    #     dicts.append({'text_id': str_hash(seq), 'text': seq, 'label': label, 'slabel': label_iit_to_slabel(label, seq)})
    # with open(os.path.join(data_dir, 'sentences_all.json'), 'w', encoding='utf8') as fp:
    #     json.dump(dicts, fp, ensure_ascii=False, indent=4) # need prettify
    # exit()

    fn = 'sentences_all.json'
    print(f'Read {fn}...')
    with open(pathjoin(data_dir, fn), 'r', encoding='utf8') as fp:
        dicts = json.load(fp)
        dicts = [d for d in dicts if d['label']]
    seq_ids = [d['text_id'] for d in dicts]
    seqs = [d['text'] for d in dicts]
    labels = [d['label'] for d in dicts]  # label_iit
    slabels = [d['slabel'] for d in dicts]

    assert all('[' not in seq and ']' not in seq for seq in seqs), "!A seq contains '[' or ']'"
    seq_hashs = [str_hash(s) for s in seqs]
    seq_ids_set, seq_hashs_set = set(seq_ids), set(seq_hashs)
    assert len(seq_ids) == len(seq_ids_set) == len(seq_hashs) == len(seq_hashs_set), '!Hash collision occurs'

    # ============================================================ Update
    need_update = False
    for i in range(len(seqs)):
        seq, label, slabel = seqs[i], labels[i], slabels[i]
        from ruleparse import TAGS
        assert all(t1 in TAGS for _, _, t1 in label), f'!Invalid tag in seq #{seq_ids[i]}: {label}'

        # update by slabel
        seq_1, label_1 = slabel_to_seq_label_iit(slabel, to_full_label=False)
        label_1 = [list(l1) for l1 in label_1]  # [[], [], ...]
        assert seq_1 == seq  # test
        if seq_1 != seq or label_1 != label:
            s = 'seq ' if seq_1 != seq else ''
            s += 'label' if label_1 != label else ''
            print(f'\tUpdate seq #{seq_ids[i]} by slabel:', s)
            dicts[i]['text'] = seq_1
            dicts[i]['label'] = label_1
            need_update = True

    # hash update
    for d in dicts:
        tid = str_hash(d['text'])
        if tid != d['text_id']:
            print(f"Text_id (hash) update: #{d['text_id']} -> #{tid}")
            d['text_id'] = tid
            need_update = True

    if need_update:
        seq_ids = [d['text_id'] for d in dicts]
        seqs = [d['text'] for d in dicts]
        labels = [d['label'] for d in dicts]
        slabels = [d['slabel'] for d in dicts]

        sent_path = pathjoin(data_dir, 'sentences_all.json')
        sent_bak_path = pathjoin(data_dir, 'sentences_all.json.bak')
        os.replace(sent_path, sent_bak_path)
        _json_pretty_dump(dicts, sent_path)
        print('Update & backup json files successfully.')

    # ========== Return
    if early_return:
        return seqs, labels, dicts

    # ============================================================  Train/val split
    seqs, labels = shuffle(seqs, labels, random_state=random_state)
    train_split = 0.8
    n = int(len(seqs) * train_split)
    train_seqs, val_seqs = seqs[:n], seqs[n:]
    train_labels, val_labels = labels[:n], labels[n:]
    train_labels = [label_iit_to_bio(train_labels[i], train_seqs[i]) for i in range(len(train_labels))]
    val_labels = [label_iit_to_bio(val_labels[i], val_seqs[i]) for i in range(len(val_labels))]

    dict_seqs = {'train': train_seqs, 'val': val_seqs}
    dict_labels = {'train': train_labels, 'val': val_labels}

    # ============================================================  Write
    if not os.path.exists(pathjoin(data_dir, 'train')):
        os.mkdir(pathjoin(data_dir, 'train'))
        os.mkdir(pathjoin(data_dir, 'val'))

    for x in ('val', 'train'):
        for i in range(len(dict_seqs[x])):  # remove space, strip non-label; is that right?
            dict_seqs[x][i] = list(dict_seqs[x][i])
            dict_seqs[x][i], dict_labels[x][i] = clean_seq_label(dict_seqs[x][i], dict_labels[x][i])

        _write_lines(pathjoin(data_dir, x, 'sentences.txt'), dict_seqs[x], add_space=True)
        _write_lines(pathjoin(data_dir, x, 'tags.txt'), dict_labels[x], add_space=True)

        with open(pathjoin(data_dir, x, '_sentences_tags.txt'), 'w', encoding='utf8') as f:
            for seq, label in zip(dict_seqs[x], dict_labels[x]):
                # f.write(f"{'    '.join(seq)}\n{' '.join(label)}\n\n")
                assert len(seq) == len(label)
                for i in range(len(seq)):
                    f.write(f'{seq[i]}/{label[i]} ')
                f.write('\n')

    # ========== Tags
    all_tags = []
    for t in train_labels:
        all_tags.extend(t)
    for t in val_labels:
        all_tags.extend(t)
    tags = set(all_tags)
    # tags.sort(key=lambda x: x[2:] + x[:2] if len(x) > 1 else x)  # sort key: B-obj -> obj-B
    with open(data_dir + 'tags.txt', 'r') as f:
        tags_last = f.readlines()
        tags_last = [t.strip() for t in tags_last if t.strip()]
    if tags != set(tags_last):
        print('! Tag changes (re-compile parser)')
        _write_lines(data_dir + 'tags.txt', tags, add_space=False)

    print(f'Process successfully, {len(seqs)} sentences in total, {n}/{len(seqs) - n} in train/val.')
    return seqs, labels, dicts


def get_data_by_text(data_dir='../data/xiaofang/', file_name = 'sentences.txt'):
    """ 根据sentences_all.txt返回seq, labels, 根据sentences_all.txt中每一行都是slabel """
    with open(data_dir + file_name, 'r', encoding='utf8') as fp:
        slabels = fp.readlines()
    slabels = [sl.strip() for sl in slabels if sl.strip() and not sl.startswith('--') and not sl.startswith('seq  :')]
    seqs, labels = zip(*[slabel_to_seq_label_iit(sl.strip('\n')) for sl in slabels if sl.strip()])
    return seqs, labels


def seq_data_loader(dataset_name):
    if dataset_name == 'json':
        seqs, labels, _ = init_data_by_json(early_return=True)
        return zip(seqs, labels)

    elif dataset_name == 'text':
        seqs, labels = get_data_by_text()
        return zip(seqs, labels)


def clean_seq_label(seq: list, label: list):
    """label: label_bio"""
    # seq_, label_ = dict_seqs[x][i], dict_labels[x][i]
    assert len(seq) == len(label)
    seq_label = list(zip(seq, label))

    for i, (s, l) in enumerate(seq_label):  # remove space
        if not s.strip():
            if l[0] == 'B' and i < len(seq_label) - 1 and l[1:] == seq_label[i + 1][1][1:]:
                seq_label[i + 1] = (seq_label[i + 1][0], l)
            seq_label[i] = None

    for i in range(len(seq_label)):  # left strip
        if seq_label[i] is None:
            continue
        if seq_label[i][1] == 'O':
            seq_label[i] = None
        else:
            break

    for i in range(len(seq_label) - 1, -1, -1):  # right strip
        if seq_label[i] is None:
            continue
        if seq_label[i][1] == 'O':
            seq_label[i] = None
        else:
            break

    seq_label = [sl for sl in seq_label if sl]
    seq, label = list(zip(*seq_label))

    assert len(seq) == len(label)

    if label[0][0] == 'I':
        label = list(label)  # tuple -> list
        label[0] = 'B' + label[0][1:]

    return seq, label


# ========================================================================================= Seq-label
def label_iit_to_bio(label_iit, seq):
    """ use BIO tags
    :param label_iit: e.g., [[9, 11, 'sobj'], [11, 14, 'obj'], [18, 22, 'prop'], [22, 26, 'cmp'], [26, 29, 'Rprop']]
    :return:  e.g., ['O',...,'B-obj','I-obj',...]
    """
    seq_len = len(seq)
    label_bio = ['O'] * seq_len
    for t in label_iit:
        i, j = t[0], t[1]  # start/end index
        if j > seq_len:
            j = seq_len
            print(f'!Index Error (ignore): get {t} in len={seq_len} sentence "{seq}"')

        label_bio[i] = 'B-' + t[2]
        for k in range(i + 1, j):
            label_bio[k] = 'I-' + t[2]

    return label_bio


def label_bio_to_iit(label_bio, seq=None):
    """
    :param label_bio: e.g., ['O',...,'B-obj','I-obj',...]
    :param seq: e.g.,['示','例','句','子'], use for checking & warning
    :return:  e.g.,[[9, 11, 'sobj'], [11, 14, 'obj'], [18, 22, 'prop'], [22, 26, 'cmp'], [26, 29, 'Rprop']]
    """
    if seq:
        # check
        if len(label_bio) != len(seq):
            print(f'!Error: label/seq len mismatch ({len(label_bio)}/{len(seq)}), label: {label_bio}, seq: {seq}')
            if len(label_bio) > len(seq):
                label_bio = label_bio[:len(seq)]
            elif len(label_bio) < len(seq):
                pass

    label_iit = []
    i = 0
    while i < len(label_bio):
        if label_bio[i].startswith('B-'):
            t = [i, i + 1, label_bio[i][2:]]
            i += 1
            while i < len(label_bio) and label_bio[i].startswith('I-'):
                t[1] += 1
                i += 1
            label_iit.append(t)
        else:
            i += 1

    return label_iit


def label_iit_to_wt(label_iit, seq, to_full_label=True):
    """label_wt (word-tag): [(words, tag),(words,tag),...] (for nltk) """
    label_iit.sort(key=lambda t: t[0])

    if to_full_label:
        label_iit = get_full_label_iit(label_iit, seq)

    flabel_wt = [(seq[i:j], t) for i, j, t in label_iit]
    return flabel_wt


def label_wt_to_iit(flabel_wt, seq_=None, to_full_label=True):
    ws, ts = list(zip(*flabel_wt))
    seq = ''.join(ws)
    if seq_:
        assert seq == seq_

    label_iit = flabel_wt.copy()
    for k in range(len(label_iit)):
        w, t = label_iit[k]
        i = label_iit[k - 1][1] if k > 0 else 0
        j = i + len(w)
        label_iit[k] = (i, j, t)

    assert all(seq[i1:j1] == flabel_wt[i][0] for i, (i1, j1, t) in enumerate(label_iit))

    if not to_full_label:
        for i in range(len(label_iit) - 1, -1, -1):
            if label_iit[i][2] == 'O':
                del label_iit[i]

    if seq_:
        return label_iit
    else:
        return label_iit, seq


def label_iit_to_slabel(label_iit, seq):
    """ input label is label_iit [(i,j,tag),..]
    slabel: combine seq and label together, e.g
            [不直度/obj]和[失圆度/obj]的[允许偏差/prop][不应大于/cmp][8mm/Rprop]
    """
    label_wt = label_iit_to_wt(label_iit, seq)
    return label_wt_to_slabel(label_wt)


def label_wt_to_slabel(label_wt):
    """ slabel: '[不直度/obj]和[失圆度/obj]的[允许偏差/prop][不应大于/cmp][8mm/Rprop]' """

    seq = []
    for word, tag in label_wt:
        if tag != 'O':
            seq.append(f'[{word}/{tag}]')
        else:
            seq.append(word)

    seq = ''.join(seq)
    return seq


def slabel_to_label_wt(slabel: str, to_full_label=True):
    """ slabel: '[贯穿孔口/obj]的[防火封堵/prop]应[直径/prop][不大于/cmp]'
        to_full_label: contains [word/O] """

    label_wt = []

    pattern = r'\[.+?/[A-Za-z]+?\]'  # [word/tag], non-greedy
    match = re.search(pattern, slabel)
    while match:
        i, j = match.span()
        if i > 0:
            if to_full_label:
                s = slabel[:i]
                if s:
                    label_wt.append((s, 'O'))
            slabel = slabel[i:]
            j, i = j - i, 0

        # i=0
        s = slabel[1:j - 1]
        idx = s.rindex('/')
        w, t = s[:idx], s[idx + 1:]
        label_wt.append((w, t))

        # finally
        slabel = slabel[j:]
        match = re.search(pattern, slabel)

    if slabel and to_full_label:
        if slabel:
            label_wt.append((slabel, 'O'))

    return label_wt


def slabel_to_seq_label_iit(slabel: str, to_full_label=True):
    flabel_wt = slabel_to_label_wt(slabel, True)
    label_iit, seq = label_wt_to_iit(flabel_wt, to_full_label=to_full_label)

    return seq, label_iit


def get_full_label_iit(label_iit, seq):
    """ full_label_iit: [((i1,j1),tag),((i2,j2),tag),...], and j(k)=i(k+1) """

    flabel_iit = [(-1, 0, 'O')]
    for i, j, tag in label_iit:
        i1, j1, _ = flabel_iit[-1]
        if i != j1:
            flabel_iit.append((j1, i, 'O'))
        flabel_iit.append((i, j, tag))
    del flabel_iit[0]

    sql = len(seq)
    if flabel_iit[-1][1] != sql:
        flabel_iit.append((flabel_iit[-1][1], sql, 'O'))
        # print(f'***Full label add right O, seq: {seq}')

    return flabel_iit


def _measure_sentence_distribution():
    def _get_doc_index(seq1):
        for i, doc in enumerate(docs):
            if seq1 in docs[i]:
                return i

        if len(seq1) <= 20:
            print(f'  Cannot find seq: {seq1}')
            return -1
        else:
            if len(seq1) % 2:
                return _get_doc_index(seq1[:-5])
            else:
                return _get_doc_index(seq1[5:])

    doc_sep = '\n\n#=====\n\n'
    with open(r'C:\Users\Zhou Yucheng\硕士\Researches\自动合规性审查\Data\xiaofang\processed\xiaofang.txt', 'r',
              encoding='utf8') as fp:
        docs = fp.read().split(doc_sep)
    for i, doc in enumerate(docs):
        d = ' '.join(doc.split('\n'))
        d = d.replace('0 . ', '0.')
        docs[i] = d

    seqs, labels, dicts = init_data_by_json(early_return=True)

    print('Calculate doc index...')
    idxs = [_get_doc_index(seq) for seq in seqs]
    print(idxs)
    from collections import Counter
    counts = Counter(idxs)
    print(counts)
    print(f'Doc count: {len(counts)}')
    return seqs, idxs, counts


def _test_random_state():
    msgs = ''
    for rs in range(10):
        init_data_by_json(random_state=rs)
        print(f'\nChecking result ... (random_state={rs})')
        corpus = Corpus('../data/xiaofang', 125)
        check, tags_df = corpus.check_tags_stratify(print_result=True)
        print('\n\n')
        r_min, r_max = min(tags_df['train-val-ratio']), max(tags_df['train-val-ratio'])
        if r_min == 0 or str(r_max) == 'inf':
            continue
        r_mm = r_max / r_min
        m = f"{rs}\t{r_min} \t{r_max} \t{r_mm:.3f}\n"
        msgs += m

    print('random states have non-inf ratios:')
    print(msgs)


if __name__ == '__main__':
    # _measure_sentence_distribution()  # 44 docs
    # _test_random_state()
    # sys.exit()

    init_data_by_json()
    check_result = True
    if check_result:
        print('\nChecking result ...')
        corpus = Corpus('../data/xiaofang', 125)
        check, tags_df = corpus.check_tags_stratify(print_result=True)
        assert check, 'check failed, consider change random_state in init_data_by_json'

"""
-Sequence and label
seq:            str of a sequence (sentence)
label_iit:      [(i,j,tag), (i,j,tag),...]
label_bio:      ['O',...,'B-obj','I-obj',...]
label_wt:       [(word,tag),(word,tag),(word,tag),..]
slabel:         str of seq & label (combined): '[word/tag][word/tag]xxx[word/tag]...'

full_label_:    contains the whole seq including O words, e.g, in label_iit, j(k)=i(k+1)
"""
