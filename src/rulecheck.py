# coding=utf-8

import sys
import argparse
import nltk
import re
import hashlib
import logging
import subprocess
import pyperclip
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple
from antlr4parser import *
from utils import *
from data import *

TAGS = {'sobj', 'obj', 'prop', 'cmp', 'Robj', 'Rprop', 'aRobj', 'aRprop'}
DEFAULT_TAG_VALUES = {'prop': 'Type', 'cmp': '='}

CMP_DICT = OrderedDict([('≤', '小于等于 小于或等于 不大于 不高于 不多于 不超过'.split()),
                        ('≥', '大于等于 大于或等于 不小于 不低于 不少于'.split()),
                        ('>', '大于 超过 高于'.split()),
                        ('<', '小于 低于'.split()),
                        ('≠', '不等于 避免 不采用 无法采用'.split()),
                        ('=', '等于 为 采用 用 按照 按 符合 执行'.split()),
                        ('has no', '不有 不设置 不设 不具备 无'.split()),
                        ('has', '有 具有 含有 留有 设 设置 设有 增设 铺设 搭设 安装 具备'.split()),
                        ('not in', '不在'.split()),
                        ('in', '在'.split()),
                        ])
# cmp reverse: 精度不低于?
# cmp improve: 无法采用? 增设?
DEONTIC_WORDS = ('应当', '应该', '应按', '应能', '尚应', '应', '必须', '尽量', '要', '宜', '得')  # '得' 通常在 '不得' 中使用


def get_cmp_str(cmp_):
    cmp_value = cmp_.values if isinstance(cmp_, RCNode) else cmp_

    if not cmp_value:  # None or ''
        return DEFAULT_TAG_VALUES['cmp']

    for dw in DEONTIC_WORDS:
        cmp_value = cmp_value.replace(dw, '')

    # simplify cmp_value, if can
    if cmp_value == '':
        return '='

    for key, values in CMP_DICT.items():
        if cmp_value in values:
            return key

    return cmp_value


def classes_same(ws: list):
    """two word is in the same class, e.g., 金属管道/铸铁管道 都属于管道
    it should be the work of ontology, and now this is a rough implementation"""
    if len(ws) <= 1:
        return True

    return all(ws[i][-2:] == ws[i + 1][-2:] for i in range(len(ws) - 1))


class LabelWordTags:
    """Label_wt, e.g., [(word, tag),(word,tag),...], nltk-friendly"""

    def __init__(self, word_tags):
        if isinstance(word_tags, nltk.Tree):
            word_tags = list(word_tags)

        # empty list, or list of word-tags
        assert isinstance(word_tags, list) and (not word_tags or (
                isinstance(word_tags[0], tuple) and len(word_tags[0]) == 2)), 'Not acceptable word-tags'

        self.word_tags = word_tags

    @property
    def tags(self):
        return [t for w, t in self.word_tags]

    def insert(self, idx, wt):
        assert isinstance(wt, tuple) and len(wt) == 2
        self.word_tags.insert(idx, wt)

    def remove(self, items):
        """
        remove/del item(s) by idx/word/(word,tag)
        :param items: idx/word/(word, tag), or a list of them
        """

        def _del_item(x1):
            # x1_: idx | word | (word, tag)
            if isinstance(x1, int):
                del self.word_tags[x1]
                return
            elif isinstance(x1, str):
                for i in range(len(self.word_tags)):
                    if self.word_tags[i][0] == x1:
                        del self.word_tags[i]
                        return
                return
            elif isinstance(x1, tuple):
                for i in range(len(self.word_tags)):
                    if self.word_tags[i] == x1:
                        del self.word_tags[i]
                        return
                return
            else:
                raise NotImplementedError

        if not items:
            return

        if isinstance(items, LabelWordTags):
            items = items.word_tags
        if not isinstance(items, list):
            items = [items]
        if isinstance(items[0], int):
            items.sort(reverse=True)  # in-place sort, descending

        for item in items:
            _del_item(item)

    def rename(self, idx, word=None, tag=None):
        if word is None:
            word = self.word_tags[idx][0]
        if tag is None:
            tag = self.word_tags[idx][1]

        self.word_tags[idx] = (word, tag)

    def append(self, wt):
        assert isinstance(wt, tuple) and len(wt) == 2
        self.word_tags.append(wt)

    def index(self, wt, default=-1):
        if isinstance(wt, list) and isinstance(wt[0], tuple) and len(wt[0]) == 2:  # list of wt
            wt = LabelWordTags(wt)

        if isinstance(wt, LabelWordTags):
            n = len(wt)
            # assert n >= 2, 'len should >=2, otherwise index may not correct'
            for i in range(len(self) - n + 1):
                if wt == self[i:i + n]:
                    return i
        else:
            for i, wt0 in enumerate(self.word_tags):
                if wt == wt0:
                    return i

        if default == -2:
            default = len(self.word_tags)
        return default

    def switch(self, idx1, idx2):
        self.word_tags[idx1], self.word_tags[idx2] = self.word_tags[idx2], self.word_tags[idx1]

    def bool_merge(self, sub_wts, idx_start=None, idx_end=None):
        """ Merge label[i_s:i_e] to one, by OR/AND. If sub_wts is given, other args will be ignored.
        Example1: [(a, obj), (or, OR), (c, obj)] -> [(a|c, obj)]
        Example2: [(a, obj), (and, AND), (c, obj)] -> [(a&c, obj)]
        Example:
            x = LabelWordTags([('a', 'obj'), ('or', 'OR'), ('c', 'obj')])
            x.bool_merge(None, 0, 3)
        """

        def get_sub_idx(x: list, y: list):
            l1, l2 = len(x), len(y)
            for i in range(l1):
                if x[i:i + l2] == y:
                    return i

        if sub_wts is not None:
            if isinstance(sub_wts, LabelWordTags):
                sub_wts = sub_wts.word_tags
            idx_start = get_sub_idx(self.word_tags, sub_wts)
            idx_end = idx_start + len(sub_wts)

        if idx_end - idx_start < 3:
            return
        wts_0 = self.word_tags[:idx_start]
        wts_m = self.word_tags[idx_start:idx_end]
        wts_1 = self.word_tags[idx_end:]

        # assert all(wts_m[i][1] == wts_m[i + 2][1] for i in range(len(wts_m) - 2)), \
        #     'Tags for bool merge should be consistent'
        if not all(wts_m[i][1] == wts_m[i + 2][1] for i in range(len(wts_m) - 2)):
            return

        t_m = wts_m[0][1]
        ws_m = []
        for w, t in wts_m:
            # if t == 'AND':
            #     ws_m.append('&')
            if t == 'OR':
                ws_m.append('|')
            else:
                ws_m.append(w)
        wt_m = [(''.join(ws_m), t_m)]

        self.word_tags = wts_0 + wt_m + wts_1

    def pop_by_tag(self, tag, remove=True):
        if all(tag != t for w, t in self.word_tags):
            return (None, tag)

        wt = next((wt for wt in self.word_tags if wt[1] == tag))  # first
        if remove:
            self.word_tags.remove(wt)
        return wt

    def tag_idxs_words(self, tag):
        i_tags_ = [i for i in range(len(self.word_tags)) if self.word_tags[i][1] == tag]
        w_tags_ = [self.word_tags[i][0] for i in i_tags_]
        return i_tags_, w_tags_

    def contains_tags(self, tags, only=False):
        if not (isinstance(tags, list) or isinstance(tags, tuple)):
            tags = [tags]

        if only:
            return all([t in tags for t in self.tags])
        else:
            return any([t in tags for t in self.tags])

    def count_tag(self, tag):
        count = 0
        for t in self.tags:
            if t == tag:
                count += 1
        return count

    def remove_tag(self, tag='O'):
        for i in range(len(self.word_tags) - 1, -1, -1):
            if self.word_tags[i][1] == tag:
                del self.word_tags[i]

    def remove_o_word(self, word='', word_len=None):
        """remove meaningless words, if its tag='O'
            if word_len is provided, del word less than word_len, and word should be empty"""
        assert bool(word) ^ bool(word_len)  # XOR

        for i in range(len(self.word_tags) - 1, -1, -1):
            if self.word_tags[i][1] == 'O':
                if word_len is None:
                    flag = self.word_tags[i][0] == word
                else:
                    flag = len(self.word_tags[i][0]) <= word_len

                if flag:
                    del self.word_tags[i]

    def hashtag(self):
        seq_ = []
        for word, tag in self.word_tags:
            seq_.append(f'{word}#{tag}')

        seq_ = ' '.join(seq_)
        return md5hash(seq_)

    def copy(self):
        return LabelWordTags(self.word_tags.copy())

    def __len__(self):
        return len(self.word_tags)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.step:  # it should be None
                raise NotImplementedError
            return LabelWordTags(self.word_tags[idx.start:idx.stop])
        else:
            return self.word_tags[idx]

    def __iter__(self):
        self.iter_idx = -1
        return self

    def __next__(self):
        self.iter_idx += 1
        if self.iter_idx == len(self.word_tags):
            raise StopIteration
        else:
            return self.word_tags[self.iter_idx]

    def __eq__(self, other):
        return self.word_tags == other.word_tags

    def __str__(self):
        """str like '[不直度/obj]_和_[失圆度/obj]_的_[允许偏差/prop]_[不应大于/cmp]_[8mm/Rprop]' """
        return label_wt_to_slabel(self.word_tags)


class RCTreeVisitor(RuleCheckTreeVisitor):
    def __init__(self):
        self.all_wts = []  # [(w,t),...]
        self.sprop_nodes = []

    def to_rcnode(self, terminal_node, default_tag=None):
        if terminal_node is None:
            assert default_tag is not None
            return RCNode(None, default_tag)

        s = str(terminal_node.getText())
        assert s[0] == '[' and s[-1] == ']'
        s = s[1:-1]

        idx = s.rfind('/')
        w, t = s[:idx], s[idx + 1:]
        if default_tag is not None:
            assert t == default_tag

        self.all_wts.append((w, t))
        return RCNode(w, t)

    def visitPrs(self, ctx: RuleCheckTreeParser.PrsContext):
        sprop_nodes = ctx.PROP()
        prop_node_ = self.visit(ctx.pr()) if (ctx.pr() is not None) else None
        sreq = self.visit(ctx.req()) if (ctx.req() is not None) else None

        assert not (sprop_nodes and sreq)  # at least one of them should be None
        if sprop_nodes:
            sprop_node = self.to_rcnode(sprop_nodes[0])

            curr_node = sprop_node
            if len(sprop_nodes) > 1:
                for i in range(1, len(sprop_nodes)):
                    curr_node.add_child(self.to_rcnode(sprop_nodes[i]))
                    curr_node = curr_node.child_nodes[0]

            if prop_node_:
                curr_node.add_child(prop_node_)

            self.sprop_nodes.append(sprop_node)
        elif sreq:
            # ()] -> ()[]
            sp = RCNode(None, 'prop')
            sp.set_req(sreq)
            self.sprop_nodes.append(sp)
            if prop_node_:
                self.sprop_nodes.append(prop_node_)
        else:
            if prop_node_:
                self.sprop_nodes.append(prop_node_)

    def visitPr(self, ctx: RuleCheckTreeParser.PrContext):
        """ (p, c, r, ro) -> RCNode-prop-req """
        prop_node = self.to_rcnode(ctx.PROP(), 'prop')
        req = self.visit(ctx.req())
        prop_node.set_req(req)

        for pr in ctx.pr():
            prop_node.add_child(self.visit(pr))

        return prop_node

    def visitReq(self, ctx: RuleCheckTreeParser.ReqContext):
        cmp = self.to_rcnode(ctx.CMP(), 'cmp')
        rprop = self.to_rcnode(ctx.RPROP())  # aR* or R*
        robj = self.to_rcnode(ctx.ROBJ()) if (ctx.ROBJ() is not None) else None  # 'Robj'

        return cmp, rprop, robj


class RCTree:
    def __init__(self, seq, label_iit):
        """
        :param seq:     sentence, list of char: ['a','b','c',...] -> str: 'abc..'
        :param label_iit:   label_tuple, [(i,j,tag),(i,j,tag),...]
        """
        self.root = RCNode('#', None)
        self.curr_node = self.root  # the node who just add_child
        self.obj_node = None  # shortcut to access obj in the tree

        self.seq = seq
        self.seq_id = md5hash(self.seq)
        self.label_iit = label_iit  # [(i,j,tag),(i,j,tag),...]
        # self.strip_seq_label()
        self.full_label = self.get_full_label()  # flabel_wt: LabelWordTags
        self.slabel = str(self.full_label)
        self.labelx = None  # self.full_label after pre_process

    def strip_seq_label(self):
        self.label_iit.sort(key=lambda t: t[0])
        start_i, end_i = self.label_iit[0][0], self.label_iit[-1][1]
        if end_i < len(self.seq):
            self.seq = self.seq[:end_i]

        if start_i > 0:
            self.seq = self.seq[start_i:]
            self.label_iit = [(i - start_i, j - start_i, t) for i, j, t in self.label_iit]

    def get_full_label(self):
        """ full_label (word-tag): [(words, tag),(words,tag),...] (for nltk)"""

        # flabel_iit = get_full_label_iit(self.label_iit)
        flabel_wt = label_iit_to_wt(self.label_iit, self.seq)

        return LabelWordTags(flabel_wt)

    def add_curr_child(self, node):
        self.curr_node.add_child(node)
        self.curr_node = node

    def pre_process1(self):
        # ================================================================================ Bool merge (union)
        def _is_union(i_, w_):
            union_words1 = {'、', '或'}
            union_words2 = {'且', '和', '及', '以及'}
            union_words3 = {',', '，'}
            # re.search('(?<!abc)x', 'abcx')
            if any(x in w_ for x in union_words1) and ('之间' not in w_):
                return True

            if any(x == w_ for x in union_words2):
                return True

            im = len(self.full_label) - 1
            if i <= 0 or i >= im:
                return False
            w_1, t_1 = self.full_label[i_ + 1]
            w1_, t1_ = self.full_label[i_ - 1]
            w_2, t_2 = self.full_label[i_ + 2] if i_ <= im - 2 else (None, None)
            w_3, t_3 = self.full_label[i_ + 3] if i_ <= im - 3 else (None, None)
            w2_, t2_ = self.full_label[i_ - 2] if i_ >= 2 else (None, None)
            if any(x == w_ for x in union_words3) and (
                    any(x == w_2 for x in union_words3) or any(x == w2_ for x in union_words3)):
                return True

            if w_ == '与' and t_1 == t1_ == 'prop' and '距' not in w_2 and '距' not in w_3:
                return True

            return False

        def _is_in_except(w_):
            return any(x in w_ for x in {':', '：'}) or len(w_) > 25

        def _is_intersection(i_):
            # 在 a 和 b 条件下 (此时是intersection)
            # a 和/或 b 之间距离 (非union)
            if i_ < len(self.full_label) - 2:
                w_2, t_2 = self.full_label[i_ + 2]
                if i_ < len(self.full_label) - 3:
                    w_2 += self.full_label[i_ + 3][0]
                return any(x in w_2 for x in {'条件下', '之间距离', '之间的距离'})
            else:
                return False

        for i, (w, t) in enumerate(self.full_label):
            if t == 'O' and _is_union(i, w) and not _is_in_except(w) and not _is_intersection(i):
                self.full_label.rename(i, tag='OR')

        for tag1 in TAGS - {'cmp'}:
            result3 = self.regex_parse(f'X: {{<{tag1}><OR><{tag1}><OR><{tag1}><OR><{tag1}>}}')
            [self.full_label.bool_merge(wts) for wts in result3]
            result2 = self.regex_parse(f'X: {{<{tag1}><OR><{tag1}><OR><{tag1}>}}')
            [self.full_label.bool_merge(wts) for wts in result2]
            result1 = self.regex_parse(f'X: {{<{tag1}><OR><{tag1}>}}')
            [self.full_label.bool_merge(wts) for wts in result1]

        for i, (w, t) in enumerate(self.full_label):
            if t == 'OR':
                self.full_label.rename(i, tag='O')

        # ============================================================ Eliminate meaningless words and duplicated O-tag
        for i in range(len(self.full_label) - 1, -1, -1):  # strip space
            w, t = self.full_label[i]
            w = w.strip()
            if not w:
                self.full_label.remove(i)
            else:
                self.full_label.rename(i, w.strip())
        self.full_label.remove_o_word('的')  # remove meaningless word
        self.full_label.remove_o_word('等')

        for i in range(len(self.full_label) - 1, 0, -1):
            (w0, t0), (w1, t1) = self.full_label[i - 1:i + 1].word_tags
            if t0 == 'O' and t1 == 'O':
                self.full_label.rename(i, w0 + ' ' + w1)
                self.full_label.remove(i + 1)

    def pre_process2(self):
        def _enum_full_label(n=2, reverse=False):
            """ for i,((w0,t0),(w1,t1)) in _enum_full_label(2) """
            if not reverse:
                range_list_ = list(range(len(self.full_label) - (n - 1)))
            else:
                range_list_ = list(range(len(self.full_label) - n, -1, -1))

            for i0_ in range_list_:
                wts_ = [self.full_label[i0_]]
                if n >= 2:
                    wts_.append(self.full_label[i0_ + 1])
                if n >= 3:
                    wts_.append(self.full_label[i0_ + 2])
                if n >= 4:
                    wts_.append(self.full_label[i0_ + 3])
                if n >= 5:
                    raise NotImplementedError
                yield i0_, wts_

        def _is_cmp_has(w_, t_):
            return t_ == 'cmp' and get_cmp_str(w_) == 'has'

        def _is_cmps(w_, t_):
            return t_ == 'cmp' or (t_ == 'O' and get_cmp_str(w_) in CMP_DICT)

        def _switch_full_label_s(i1_, i2_):
            """_switch_full_label_sequentially"""
            assert i2_ > i1_
            print(f'[DEBUG] switch p-r tag: ({self.full_label[i1_ - 1:i2_ + 1]})', end=' -> ')
            for i_ in range(i2_ - 1, i1_ - 1, -1):
                self.full_label.switch(i_, i_ + 1)
            print(self.full_label[i1_ - 1:i2_ + 1])

        # ================================================================================ Sub-seq switch
        # 1) A 应符合下列规定： B 时 C -> B A C
        def _sub_seq_switch(i, j):
            if i < j < len(self.full_label) - 1:
                print(f'[DEBUG] sub-seq switch at {self.full_label[i:j]}')
                ar_wts = self.full_label.word_tags[i:j]  # aR
                self.full_label.remove(list(range(i, j)))
                self.full_label.word_tags += ar_wts  # TODO 对ar_wts单独生成一个sub-tree

        for i, (wi, ti) in enumerate(self.full_label):
            if ti == 'O':
                js = [j for j, (wj, tj) in enumerate(self.full_label[i + 1:]) if tj == 'O' and '时' in wj]
                if js and (re.search(r'^(应符合|不应.于)下列规定：', wi) or wi.strip(' ,，') in ('在', '当')):
                    j = i + 1 + js[0]
                    has_arprop = any(t == 'aRprop' for w, t in self.full_label[i + 1:j])
                    has_no_obj = all(t != 'obj' for w, t in self.full_label[i + 1:j])
                    if has_arprop and has_no_obj:
                        j = i + 1 + js[0]
                        _sub_seq_switch(i, j)
                        break
                if re.search(r'^(应符合|不应.于)下列规定：', wi):
                    lwtss = self.regex_parse('X: {<O><.*>*<aRprop><O>}', self.full_label[i:])  # greedy
                    if len(lwtss) == 1:
                        lwts = lwtss[0]
                        j = i + len(lwts) - 1  # tag-j is O
                        _sub_seq_switch(i, j)
                        break

        # 2) 当A被B C时 -> 当B把A C时
        for i, lwts in self.regex_parse('X: {<a?Robj>?<a?Rprop><O><obj><prop>}', return_idx=True):
            if i >= 1 and '当' in self.full_label[i - 1][0] and self.full_label[i + len(lwts) - 3][0] == '被':
                self.full_label.rename(i + len(lwts) - 3, '把')
                _switch_full_label_s(i, i + len(lwts) - 2)
                _switch_full_label_s(i + 1, i + len(lwts) - 2)

        # ================================================================================ Has prop
        # 1.1) has p O -> has Rp p O
        for lwts in self.regex_parse('X: {<cmp><prop><O>}'):
            if _is_cmp_has(*lwts[0]):
                i = self.full_label.index(lwts)
                if i >= 0:
                    self.full_label.insert(i + 1, (', ', 'O'))  # avoid p-r swtich here
                    self.full_label.insert(i + 1, (lwts[1][0], 'Rprop'))
                    print(f'[DEBUG] (has prop) insert Rprop in {self.full_label[i:i + 3]}')

        # 2.1) has R 其 -> has R p
        for lwts in self.regex_parse('X: {<cmp><Rprop><O|prop>}'):
            if _is_cmp_has(*lwts[0]) and '其' in lwts[2][0]:
                i = self.full_label.index(lwts)
                if i >= 0:
                    self.full_label.insert(i + 3, (lwts[1][0], 'prop'))
                    print(f'[DEBUG] (has prop) insert prop in {self.full_label[i:i + 4]}')

        # 2.2) has c R P -> has P c R -> has Rp, p c R
        for i, lwts in self.regex_parse('X: {<cmp><cmp><a?Robj>?<a?Rprop><prop>}', return_idx=True):
            if _is_cmp_has(*lwts[0]):
                _switch_full_label_s(i + 1, i + len(lwts) - 1)
                self.full_label.insert(i + 1, (', ', 'O'))  # avoid p-r swtich here
                self.full_label.insert(i + 1, (lwts[-1][0], lwts[-2][1]))

        # 3.0) has c? R1 p1 R2 -> has p1 c? R1 R2 (p-r switch for 3.1)
        for i, lwts in self.regex_parse('X: {<cmp><cmp>?<Rprop><prop><Rprop>}', return_idx=True):
            if _is_cmp_has(*lwts[0]):
                _switch_full_label_s(i + 1, i + len(lwts) - 2)

        # 3.1) has/eq p1 c? R p2 -> [has Rp2,] p2 p1 c? R
        for i, lwts in self.regex_parse('X: {<cmp><prop><cmp>?<a?Robj>?<a?Rprop><R?prop>}', return_idx=True):
            # while lwts := regex()
            c_str = get_cmp_str(lwts[0][0])
            if c_str in ('has', '='):
                self.full_label.rename(i + len(lwts) - 1, tag='prop')
                _switch_full_label_s(i + 1, i + len(lwts) - 1)
                if c_str == 'has':
                    self.full_label.insert(i + 1, (', ', 'O'))  # avoid p-r swtich here
                    self.full_label.insert(i + 1, (lwts[-1][0], lwts[-2][1]))  # can be aR OR R
                else:
                    self.full_label.remove(i)
                print(f'[DEBUG] (has prop) p-r switch and insert Rprop in {self.full_label[i:i + 3]}')

        # ================================================================================ Prop-Rprop (p-r) switch
        for w1 in ('作', '作为', '进行', '处'):
            self.full_label.remove_o_word(w1)

        # ========== (c ro r) order
        # 0.1) ro c r -> c ro r
        for i, lwts in self.regex_parse('X: {<a?Robj><cmp><a?Rprop>}', return_idx=True):
            _switch_full_label_s(i, i + 1)

        # 0.2) c r ro -> c ro r
        for i, lwts in self.regex_parse('X: {<cmp><a?Rprop><a?Robj>}', return_idx=True):
            if i + 2 == len(self.full_label) - 1 or not self.full_label[i + 3][1].endswith('Rprop'):
                _switch_full_label_s(i + 1, i + 2)

        # 0.3) p c ro O -> p c ro p O
        for i, lwts in self.regex_parse('X: {<prop><cmp><Robj>}', return_idx=True):
            if not any(t == 'Rprop' for w, t in self.full_label[i + 2:i + 5]):
                self.full_label.insert(i + 3, (lwts[0][0], 'Rprop'))

        # ========== C r p
        # 1.1) C r p (Note: 1.x cannot use self.regex_parse, because it may return over-lapping word-tag, cannot update)
        for i, ((w0, t0), (w1, t1), (w2, t2)) in _enum_full_label(3, reverse=True):
            if _is_cmps(w0, t0) and t1.endswith('Rprop') and t2 == 'prop':
                _switch_full_label_s(i, i + 2)

        # 1.2) C ro r p
        for i, ((w0, t0), (w1, t1), (w2, t2), (w3, t3)) in _enum_full_label(4, reverse=True):
            if _is_cmps(w0, t0) and t1.endswith('Robj') and t2.endswith('Rprop') and t3 == 'prop':
                _switch_full_label_s(i, i + 3)

        # ========== r p
        # 2.1) AND r p
        for i, lwts in self.regex_parse('X: {<O><Robj>?<Rprop><prop>}', return_idx=True):
            if lwts[0][0] in ('和', '或', '并'):
                _switch_full_label_s(i + 1, i + len(lwts) - 1)

        # 2.2) r p O
        for i, lwts in self.regex_parse('X: {<Robj>?<Rprop><prop><O>}', return_idx=True):
            _switch_full_label_s(i, i + len(lwts) - 2)

        # 2.3) ^r p
        for i, lwts in self.regex_parse('X: {^<O|obj>*<a?Robj>?<a?Rprop><prop><O>}', return_idx=True):
            j = int(lwts.contains_tags('aRobj') or lwts.contains_tags('Robj'))  # 0 or 1
            _switch_full_label_s(i + len(lwts) - 3 - j, i + len(lwts) - 2)

        # TODO p1 p2 p3 xxx R, match p2-R not p1-R
        # ### px p r p
        # for i, ((w0, t0), (w1, t1), (w2, t2), (w3, t3)) in _enum_full_label(4, reverse=True):
        #     if t0 == 'prop' and t1 == 'cmp' and t2.endswith('Rprop') and t3 == 'prop':
        #         _switch_full_label_s(i, i + 3)

        # ================================================================================ Match prop
        while self.regex_parse('X: {<prop><cmp><Robj>?<Rprop><O><cmp><Rprop>}', return_idx=True):
            i, lwts = self.regex_parse('X: {<prop><cmp><Robj>?<Rprop><O><cmp><Rprop>}', return_idx=True)[0]
            n = len(lwts)
            if lwts[n - 3][0].strip('，') in {'并', '且', '并且', '即'}:
                self.full_label.insert(i + n - 2, lwts[0])
                print(f"[DEBUG] pre match p-r tag for in {lwts}")
            else:
                break

        print(f'[INFO] LabelX:\t{self.full_label}\n')
        self.labelx = self.full_label.copy()
        self.full_label.remove_tag('O')

    def post_process(self):
        def _is_union_word(w_: str):
            for dw in DEONTIC_WORDS:
                w_ = w_.replace(dw, '')
            # do not use ('且', '并'): [熔点/prop]_[不小于/cmp]_[1000℃/aRprop]_且_[无/cmp]_[绝热层/aRprop]_的...
            return any(w_.endswith(x) for x in {'，并', '，且', '，并且', '，即'})

        self.obj_node: RCNode
        if not self.obj_node:
            return
        props: List[RCNode] = self.obj_node.child_nodes
        full_label_x_ = self.full_label.copy()
        self.full_label = self.get_full_label()
        self.pre_process1()

        # # ===== Sort by index in full label, then by value-tag. Bad idea, because default value
        # pis = [self.full_label.index((p.values, p.tag), -2) for p in props]
        # pstrs = [p.tree_str('-', False, True) for p in props]
        # props = [p for pi, ps, p in sorted(zip(pis, pstrs, props))]
        # self.obj_node.child_nodes = props

        # ===== Default prop, p-r match, match last prop
        for i in range(1, len(props)):  # use ascending order
            # p R 并/且|其 R
            pi = props[i]
            pi1 = props[i - 1]
            if pi is None or pi1 is None:
                continue
            is_prr = bool(pi1.values and pi1.req and not pi.values and pi.req)
            is_same_req = pi.is_app_req() == pi1.is_app_req()
            if self.seq_id == 'c0ff675':
                x = 1
            if is_prr and is_same_req:
                li1 = self.full_label.index((pi1.values, pi1.tag))  # last prop
                if li1 < 0 or li1 > len(self.full_label) - 3:
                    continue
                req = [(r.values, r.tag) for r in pi.req if r]  # if r.values
                full_label_1 = self.full_label[li1 + 1:]
                if len(req) <= 2:
                    li = li1 + 1 + full_label_1.index(req)
                else:
                    # li = the first valid index of req
                    lis = [li1 + 1 + li_ for li_ in (full_label_1.index(req),
                                                     full_label_1.index([req[0], req[2], req[1]]),
                                                     full_label_1.index(req[:2]))
                           if li_ >= 0]
                    if not lis:
                        continue
                    li = lis[0]

                # lis = [li1 + 1 + self.full_label[li1 + 1:].index((r.values, r.tag)) for r in pi.req if r] # bad idea
                is_next = (0 <= li1 < li < len(self.full_label)) and all(
                    t != 'prop' for w, t in self.full_label[li1 + 1:li])
                if is_next and _is_union_word(self.full_label[li - 1][0]):
                    pi.values = pi1.values
                    print(f"[DEBUG] Match p-r tag for {str(pi1)} in {self.full_label[li1:li + 1]}")

            elif pi1.values and pi1.child_nodes and pi.values == '其' and pi.child_nodes and is_same_req:
                print(f'[DEBUG] Match last prop1 {pi}')  # #works only for one
                pi1.child_nodes += pi.child_nodes
                props[i] = None

            elif pi1.values and pi1.child_nodes and all(pi.values == n.values for n in pi1.child_nodes):
                print(f'[DEBUG] Match last prop2 {pi}')  # #works only for one
                pi1.child_nodes.append(pi)
                props[i] = None

        for i in range(len(props) - 1, -1, -1):
            if props[i] is None:
                del props[i]

        # ===== Remove duplicated prop (recursive)
        def _rm_duplicated_child_prop(props_):
            for i_, prop_ in enumerate(props_):
                _rm_duplicated_child_prop(prop_.child_nodes)
                if prop_.n_child() == 1 and prop_.values == prop_.child_nodes[0].values and prop_.tag == \
                        prop_.child_nodes[0].tag:
                    props_[i_] = prop_.child_nodes[0]

        _rm_duplicated_child_prop(props)

        # ===== Match aRprop-obj (anchor)
        if isinstance(self.obj_node.values, list):
            anchored = False
            slabel = str(self.full_label)
            iobjs = [slabel.index(f'[{v}/obj]') for v in self.obj_node.values]
            for prop in props:
                if prop.is_app_req() and not isinstance(prop.values, list):
                    vp = prop.values if prop.values is not None else ''
                    ip = re.search(f'{vp}.*?{prop.req[1].values}/aRprop', slabel).span()[1]
                    for k, io in enumerate(iobjs):
                        if ip < io:
                            prop.anchor = f'&{k}' if ('时，' in slabel[ip:io]) else f'&{k + 1}'
                            anchored = True
                            break
                    if not prop.anchor:
                        prop.anchor = f'&{len(iobjs)}'
                        anchored = True
            if anchored:
                for i, v in enumerate(self.obj_node.values):
                    self.obj_node.values[i] += f'&{i + 1}'

        # ===== Restore self.full_label
        self.full_label = full_label_x_

    def regex_parse(self, grammar, full_label=None, return_idx=False):
        """
        :param grammar: e.g., 'P: {<prop><propx><cmp>?<a?Rpropx>}'
        :param full_label: let it None to use self.full_label
        :param return_idx: set True to return [(idx, lwts),...]
        :return: list of LabelWordTags, return an empty list when no result
        """
        if not full_label:
            full_label = self.full_label

        cp = nltk.RegexpParser(grammar)
        result = cp.parse(full_label.word_tags)
        result = [LabelWordTags(x) for x in result if isinstance(x, nltk.Tree)]

        if return_idx:
            result = [(full_label.index(lwts), lwts) for lwts in result if full_label.index(lwts) >= 0]

        return result

    def antlr4_parse(self):
        input_stream = InputStream(str(self.full_label))
        lexer = RuleCheckTreeLexer(input_stream)
        tokens = CommonTokenStream(lexer)
        parser = RuleCheckTreeParser(tokens)
        parser._listeners = [RCTreeErrorListener()]
        # parser.addErrorListener(RCTreeErrorListener())
        tree = parser.rctree()

        # log(tree.toStringTree(recog=parser))
        return tree

    def is_gen_complete(self):
        """whether consumes all available word-tags"""
        return self.full_label.contains_tags(('O', 'obj'), only=True)

    def generate(self):
        log_msg = f'[{total_count + 1}]#{self.seq_id}\n'
        log_msg += f'Seq:\t{self.seq}\n'
        log_msg += f'Label:\t{self.slabel}\n'  # slabel

        # ================================================================================ Pre-process & Obj
        self.pre_process1()

        if 'sobj' in self.full_label.tags:
            i_sobjs, w_sobjs = self.full_label.tag_idxs_words('sobj')
            if len(i_sobjs) == 1:
                self.add_curr_child(RCNode(w_sobjs[0], 'sobj'))
            else:
                while w_sobjs:
                    for i in range(len(w_sobjs), 0, -1):
                        if classes_same(w_sobjs[:i]):
                            self.add_curr_child(RCNode(w_sobjs[:i], 'sobj'))
                            w_sobjs = w_sobjs[i:]
                            break

            self.full_label.remove(i_sobjs)

        if 'obj' in self.full_label.tags:
            i_objs, w_objs = self.full_label.tag_idxs_words('obj')
            self.obj_node = RCNode(w_objs, 'obj')
            self.add_curr_child(self.obj_node)
            if len(i_objs) > 1:
                self.full_label.remove(i_objs[1:])  # leave one obj
        else:
            self.obj_node = RCNode('*', 'obj')
            self.add_curr_child(self.obj_node)

        self.pre_process2()

        # ================================================================================ Parsing generate
        try:
            self._generate_by_antlr4()
        except ParserError as ex:
            log_msg = str(ex) + '\n' + log_msg

        self.post_process()
        # ================================================================================ Finish
        is_comp = self.is_gen_complete()

        # # === Simple/Complex sentence
        # labelx = str(self.labelx)
        # n_propl = labelx.count('/prop]')
        # n_Rpropl = labelx.count('/Rprop]')
        # n_aRpropl = labelx.count('/aRprop]')
        # n_cmp = labelx.count('/cmp]')
        # rcts = [x for x in str(self).split('\n') if x]
        # is_2_props = len(rcts) - 1 > 1
        # self.is_simple = not is_2_props
        # if self.is_simple:
        #     print('***Simple', n_propl, n_Rpropl + n_aRpropl, n_cmp)
        #     assert n_propl == 1 and n_Rpropl + n_aRpropl == 1 and n_cmp == 1, self.seq
        # else:
        #     print('***Complex', n_propl, n_Rpropl + n_aRpropl)
        #     assert n_propl > 1 or n_Rpropl + n_aRpropl > 1 or n_cmp > 1, self.seq
        # return

        log_msg += f"RCTree:\t#{self.root.hashtag()}\n{self}\n"
        log_msg += 'Gen complete.\n' if is_comp else f'Gen not complete: {self.full_label}\n'
        log_msg += '-' * 90
        log(log_msg, print_log=True)  # print all, print_log=not is_comp

        return is_comp, log_msg

    def _generate_by_regex(self):
        def _add_props(labelwts_list):
            """ label_wts_list: [LabelWordTags, LabelWordTags,...], typically from self.regex_parse() """
            if not isinstance(labelwts_list, list):
                labelwts_list = [labelwts_list]

            for label_wts in labelwts_list:
                label_wts.pop_by_tag('obj')  # leave obj (if any)

                has_propx = any('propx' in t for w, t in label_wts.word_tags)
                wt = label_wts.pop_by_tag('prop', remove=has_propx)  # del prop (remove=False) when not has_propx
                prop_node = RCNode(wt[0], wt[1])  # if wt=(None, Tag), then RCNode use the default value
                if has_propx and wt[0] is not None:
                    self.del_delayed_props.append(wt)  # should delete prop delayed

                if has_propx:
                    wtx = label_wts.pop_by_tag('propx', remove=False)  # del propx
                    propx_node = RCNode(wtx[0], wtx[1])
                    propx_node.set_req_by_wts(label_wts.word_tags)
                    prop_node.add_child(propx_node)
                else:
                    prop_node.set_req_by_wts(label_wts.word_tags)

                self.obj_node.add_child(prop_node)
                self.full_label.remove(label_wts)

        def _add_props_by_regex_parsing(grammars, loop=False):
            if isinstance(grammars, list):
                for g in grammars:
                    _add_props_by_regex_parsing(g, loop)
                return

            result_last = None
            while True:
                result = self.regex_parse(grammars)
                if result:
                    _add_props(result)

                if not loop or not result or result == result_last:
                    break

                result_last = result

        self.del_delayed_props = []
        # for those who do not have <O> (parse just once)
        grammars_first = \
            ['P: {<prop><propx><cmp>?<a?Rpropx>}',
             'P: {<prop><cmp>?<a?Rprop>}',
             'P: {<aRprop><obj>}',
             'P: {<obj><aRprop>}',
             ]
        _add_props_by_regex_parsing(grammars_first)

        # self.full_label.remove_tag('O')
        # for propx
        grammar = r"""
        P: {<prop><cmp|O>*<propx><O>*<a?Rpropx>}
        P: {<prop><cmp|O>*<a?Rpropx><O>*<propx>}
        P: {<prop><O>*<propx><cmp|O>*<a?Rpropx>}
        P: {<prop><O>*<propx>?<cmp|O>*<a?Rpropx>}
        """
        _add_props_by_regex_parsing(grammar, loop=True)
        if self.full_label.contains_tags(['Rpropx', 'aRpropx']):
            log(f'!Failed to parse /propx grammar: {self.full_label}')

        # for prop
        grammar = r"""
        P: {<prop><cmp><Robj>?<O>*<a?Rprop>}
        P: {<prop><cmp|O>*<Robj>?<O>*<a?Rprop>}
        P: {<prop><cmp|O>*<Robj|O>*<a?Rprop>}
        """
        _add_props_by_regex_parsing(grammar, loop=True)
        if self.full_label.contains_tags(['Rpropx', 'aRpropx']):
            log(f'!Failed to parse /prop grammar: {self.full_label}')

        self.full_label.remove(self.del_delayed_props)

    def _generate_by_antlr4(self):
        def _add_props_by_antlr4_parsing(loop=1):
            for i in range(loop):
                tree = self.antlr4_parse()
                if not tree:
                    return

                visitor = RCTreeVisitor()
                visitor.visit(tree)
                self.full_label.remove(visitor.all_wts)
                for p_node in visitor.sprop_nodes:
                    self.obj_node.add_child(p_node)

                if self.is_gen_complete():
                    return

        # ===== First loop
        _add_props_by_antlr4_parsing(loop=2)

        return

        # # ===== Next loop
        # def _swap_tags_if_one(tag1, tag2, reserve_order=True):
        #     if self.full_label.count_tag(tag1) == self.full_label.count_tag(tag2) == 1:
        #         i1 = self.full_label.tag_idxs_words(tag1)[0][0]
        #         i2 = self.full_label.tag_idxs_words(tag2)[0][0]
        #
        #         # reserve_order means sway only when i1>i2, that is, make tag1 prior to tag2
        #         if not reserve_order or i1 > i2:
        #             self.full_label.switch(i1, i2)
        #
        # _swap_tags_if_one('prop', 'Rprop')
        # _swap_tags_if_one('prop', 'aRprop')
        #
        # _add_props_by_antlr4_parsing(loop=1)
        # if self.is_gen_complete(): return
        #
        # log(f'Cannot parse all. Full_label: {self.full_label}')

    def __str__(self, indent='\t\t'):
        """
        # -> [sobj] -> [obj]
            if:     [prop] = [Rprop]
            check:  [prop] = [Rprop]
                    [prop] = [Rprop]<-[Robj]
                    [prop]->[prop] = [Robj]<-[Rprop]
            ← →
        """

        tree = indent + str(self.root)
        if not self.root.has_child():
            return tree

        # ========== Obj (consider first child only)
        assert self.root.n_child() == 1
        node = self.root.child_nodes[0]
        while node is not self.obj_node:
            tree += f"->{str(node)}"
            assert node.n_child() == 1
            node = node.child_nodes[0]
        tree += f"->{str(self.obj_node)}"

        if not self.obj_node.has_child():
            # ignore obj's Robj now
            return tree

        # ========== Obj tree
        obj_tree = self.obj_node.tree_str()
        lines = obj_tree.split('\n')[1:]

        # ===== Sort
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].startswith('|--'):  # propx+
                lines[i - 1] += '\n' + indent + lines[i]
                del lines[i]

        idx_lines = []
        for li, l in enumerate(lines):
            is_check = not ('-?' in l)
            l1 = l[:l.index('\n')] if '\n' in l else l
            p = l1[l1.index('['):l1.index(']')] + '/'
            rp = l1[l1.rindex('['):l1.rindex(']')] + '/'
            if '|' in p:
                p = p[:p.index('|')] + '/'
            if '|' in rp:
                rp = rp[:rp.index('|')] + '/'
            pi = self.slabel.find(p)
            rpi = self.slabel.find(rp)
            if pi < 0:
                pi = len(self.slabel)
            if rpi < 0:
                rpi = len(self.slabel)
            ppi = min(pi, rpi)
            idx_lines.append((is_check, ppi, li, l))
        idx_lines.sort()
        lines = [l for *_, l in idx_lines]
        # =====

        lines = [indent + l for l in lines]
        obj_tree = '\n'.join(lines)

        return tree + '\n' + obj_tree


class RCNode:
    """The basic element in Rule Check,
    it can be an object/instance/property, or a union/intersection of them """

    def __init__(self, values, tag):
        """value: word(str) or list of word(str)"""
        if isinstance(values, list) or isinstance(values, tuple):
            values = [v for i, v in enumerate(values) if v not in values[:i]]  # remove duplications
            for i in range(len(values) - 1, 0, -1):  # remove duplications
                if values[i - 1].endswith(values[i]) and '|' not in values[i - 1]:
                    del values[i]
            if len(values) == 1:
                values = values[0]

        self.values = values
        self.tag = tag

        self.anchor = ''  # anchor to a specific obj, when there are multiple objs
        self.child_nodes = []
        self.req = None  # a tuple of (cmp_node, req_node, sreq_node=None)

    def is_app_req(self):
        if self.req:
            return self.req[1].tag[0] == 'a'
        return None

    def has_app_seq(self):
        if self.req:
            return self.req[1].tag[0] == 'a'
        elif self.child_nodes:
            return any(cn.has_app_seq() for cn in self.child_nodes)
        else:
            return False

    def add_child(self, node):
        self.child_nodes.append(node)

    def has_child(self):
        return bool(self.child_nodes)

    def n_child(self):
        return len(self.child_nodes)

    def set_req(self, req: tuple):
        assert self.req is None, 'Each RCNode has at most one req'
        assert len(req) == 3
        assert 'Rprop' in req[1].tag  # '[a]Rprop'

        self.req = req

    def set_req_by_wts(self, wts):
        """ wts: list/tuple of word-tag (<O> tags will be filtered)"""
        if self.tag == 'obj':
            ct, rt, srt = 'cmp', 'Robj', None  # may be 'Rsobj' in the future
        elif self.tag == 'prop':
            ct, rt, srt = 'cmp', 'Rprop', 'Robj'
        else:
            raise AssertionError('add_req_by_tuple() should be used for only /obj, /prop')
        if any(('aR' in wt[1] for wt in wts)):
            rt = 'a' + rt

        # Double check
        for w, t in wts:
            assert t in (ct, rt, srt, 'O') or (w, t) == (self.values, self.tag)

        cv, rv, srv = DEFAULT_TAG_VALUES[ct], None, None
        for v, t in wts:  # value, tag
            if ct == t:
                cv = v
            elif rt == t:
                rv = v
            elif srt == t:
                srv = v

        cn = RCNode(cv, ct)
        rn = RCNode(rv, rt)
        srn = RCNode(srv, srt) if srv is not None else None
        self.set_req((cn, rn, srn))

    def hashtag(self):
        all_str = self.tree_str(indent='-', optimize=False)
        return md5hash(all_str)

    def tree_str(self, indent='-', optimize=True, show_tag=False):
        all_str = self.__str__(optimize, show_tag, True)

        if self.has_child():
            cn_all_strs = []
            for cn in self.child_nodes:
                cn_all_strs.append(cn.tree_str(indent + indent[0], optimize))

            if self.n_child() >= 2:
                cns_strs = list(zip([not cn.has_app_seq() for cn in self.child_nodes], cn_all_strs))
                cns_strs.sort()
                cn_all_strs = [s for cn, s in cns_strs]

            sep = f'\n|{indent}'
            all_str += sep + sep.join(cn_all_strs)

        return all_str

    def __str__(self, optimize=True, show_tag=False, show_req=False):
        values = self.values

        if optimize:
            default_value = DEFAULT_TAG_VALUES[self.tag] if self.tag in DEFAULT_TAG_VALUES else '?'
            if not values:
                values = default_value
            elif isinstance(values, list) and None in values:
                for i in range(len(values)):
                    if values[i] is None:
                        values[i] = default_value

            if self.tag == 'cmp':
                return get_cmp_str(values)

            if self.tag == 'prop' and self.values is None:
                if self.req and str(self.req[0]).startswith('has'):
                    values = 'Props'

            # TODO default_cmp_value 高于: 位置 大于, etc

        if isinstance(values, list) or isinstance(values, tuple):
            values = ', '.join(self.values)

        str_ = f'[{values}{self.anchor}]'

        if show_tag:
            str_ += f'#{self.tag}'

        if show_req:
            if self.req:
                # req_str_: cmp, Rprop [,Robj]
                req_str_ = f"{self.req[0].__str__(optimize)} {self.req[1].__str__(optimize)}"
                if self.req[2] is not None:
                    req_str_ += f"<-{self.req[2].__str__(optimize)}"

                str_ = f"{str_} {req_str_}"

                # ? means if
                prefix = '?' if self.is_app_req() else ''
                str_ = prefix + str_
            else:
                pass

        return str_

    def __bool__(self):
        return bool(self.values)


def model_data_loader():
    seqs_raw, labels_raw, _ = init_data_by_json()
    train_data_loader, val_data_loader, corpus = get_data_loader('../data/xiaofang', batch_size=1,
                                                                 check_stratify=False, shuffle_train=False)
    log('-' * 90)

    def resolve_token(seq: list):
        assert isinstance(seq, list)
        seq_len = len(seq)
        seq_list = seq
        seq = ''.join(seq)

        if '[UNK]' in seq_list:
            # print('*** ' + seq)
            escape_chars = ['$', '(', ')', '*', '+', '.', '[', ']', '?', '\\', '^', '{', '}', '|']
            for i, s in enumerate(seq_list):
                if s in escape_chars:
                    seq_list[i] = '\\' + s
            seq_list = ''.join(seq_list)
            seq_list = seq_list.replace('[UNK]', '.')

            matches = []
            for seq_raw in seqs_raw:
                seq_raw_ = seq_raw.replace(' ', '')
                x = re.search(seq_list, seq_raw_)
                if x:
                    matches.append((x, seq_raw_))

            if len(matches) != 1:
                len_min = min([len(s) for x, s in matches])
                flag = True
                for k in range(len(matches) - 1):
                    (i, j), s = matches[k][0].span(), matches[k][1]
                    (i1, j1), s1 = matches[k + 1][0].span(), matches[k + 1][1]
                    if not s[i:j] == s1[i1:j1]:
                        flag = False
                        break
                if flag:
                    matches = [matches[0]]

            if len(matches) == 1:
                (i, j), seq_raw = matches[0][0].span(), matches[0][1]
                assert j - i == seq_len, f'!Seq len not match after resolve, seq: {seq}'
                seq = seq_raw[i:j]
                # print('*** ' + seq)
            else:
                logging.warning('Failed to find [UNK]:')

        assert '[UNK]' not in seq
        return seq

    def yield_data(_data_loader):
        for token_ids, att_mask, tag_ids in _data_loader:
            # seq: tokens, label: tags
            mask = corpus.get_token_ids_bool_mask(token_ids[0])  # filter out [PAD] tokens
            seq = corpus.tokenizer.convert_ids_to_tokens(token_ids[0].masked_select(mask))  # -> list
            label = corpus.ids_to_tags(tag_ids[0].masked_select(mask))  # -> list
            label_t = label_bio_to_iit(label, seq)

            seq = resolve_token(seq)  # -> str

            yield seq, label_t

    for seq, label_t in yield_data(val_data_loader):
        yield seq, label_t

    for seq, label_t in yield_data(train_data_loader):
        yield seq, label_t

    # seqs, labels = process_xiaofang_label_json()
    #
    # for i in range(len(seqs)):
    #     seq, label_t = seqs[i], labels[i]
    #     idx_min, idx_max = min([t[0] for t in label_t]), max([t[1] for t in label_t])
    #     if idx_min > 0:
    #         for t in label_t:
    #             t[0] -= idx_min
    #             t[1] -= idx_min
    #     if idx_max < len(seq):
    #         seqs[i] = seq[:idx_max]
    #
    #     label_bio = seq_label_tuple2bio(labels[0], seqs[0])
    #     idxs = [i for i in range(len(seq)) if i != ' ']
    #     seq = ''.join([seq[i] for i in idxs])
    #     label_bio = [label_bio[i] for i in idxs]
    #     label_t = seq_label_bio2tuple(label_bio, seq)
    #
    # return zip(seqs, labels)


def json_data_loader():
    seqs, labels, _ = init_data_by_json(return_only=True)

    # # hash collision check
    # seq_hashs = [md5hash(s) for s in seqs]
    # assert len(seq_hashs) == len(set(seq_hashs)), '! hash collision occurs, consider use a longer md5hash length'

    # json_dict = {}
    # for i in range(len(seqs)):
    #     seq = seqs[i]
    #     label_iit = labels[i]
    #     # slabel = label_iit_to_slabel(label_iit, seq)
    #     # d = {'text': seqs[i], 'label': label_iit, 'slabel': slabel}
    #     d = {'text': seqs[i], 'label': label_iit}
    #     json_dict.update({seq_hashs[i]: d})

    return zip(seqs, labels)


class EvalLogFile:
    SEP = '\n' + '-' * 90 + '\n'
    EVALs = ('##correct', '##wrong', '##relabel', '##del')

    def __init__(self, file_txt):
        self.txt = file_txt
        self.msgs = self.txt.split(self.SEP)
        self.ddict = self.get_ddict()  # {hashtag: msg-dict, ...}, use msgs[1:-1]

    def get_ddict(self):
        ddict = OrderedDict()
        for msg in self.msgs[1:-1]:
            ddict.update(self.msg2dict(msg))  # like append

        return ddict

    def msg2dict(self, msg: str):
        """
        :param msg:
            (optional line)!SyntaxError: no viable alternative at input 'xxxx'
            [70]#6bd06d8
            Sequence #6bd06d8:	天馈系统的驻波比不应大于2
            Label	 #82a8e33:	[天馈系统/obj] 的 [驻波比/prop] [不应大于/cmp] [2/Rprop]
            RCTree	 #f08b2eb
                [#]->[天馈系统]
                    check:	[驻波比] ≤ [2]
            Gen complete.
            (optional line)##correct
        :return:
        """

        # === check & pre-process, move 'SyntaxError' to the second line
        assert ']#' in msg and 'Seq' in msg
        lines = [l for l in msg.split('\n')]
        assert len(lines) > 5
        if lines[0].startswith('!SyntaxError'):
            msg = '\n'.join([lines[1]] + [lines[0]] + lines[2:])
        assert msg[0] == '[', f"msg format wrong! {msg}"

        # ==========
        lines = [l for l in msg.split('\n') if l.strip()]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('Gen complete') or lines[i].startswith('Gen not complete'):
                del lines[i]
            elif lines[i].startswith('!SyntaxError'):
                del lines[i]

        seq_id_, seq, label, rct, eval = lines[0], lines[1], lines[2], lines[3], lines[-1]

        i = seq_id_.index('#')
        idx = int(seq_id_[1:i - 1])
        seq_id_ = seq_id_[i:]
        msg = msg[i:]  # remove [idx] in msg

        assert seq.startswith('Seq:\t')
        seq = seq[seq.index(':') + 2:]

        assert label.startswith('Label:\t')
        label = label[label.index(':') + 2:]

        if eval.startswith('##'):
            eval_ = eval.replace('###', '##')
            assert any(eval_.startswith(e) for e in self.EVALs), f'invalid eval: {eval}'
        else:
            eval = ''

        rct_id_ = rct[rct.index('#'):]
        rcts = ('\n'.join(lines[4:-1])).replace('\t', '')
        assert seq_id_[0] == '#' and rct_id_[0] == '#'

        # the idx line is removed in d['msg']
        d = {'idx': idx, 'seq': seq, 'label': label, 'rct_id': rct_id_, 'rctree': rcts, 'eval': eval, 'msg': msg}
        return {seq_id_: d}

    def update_eval(self):
        msgs_idxs = []
        for seq_id, d in self.ddict.items():
            d['msg'] = f"[{d['idx']}]{d['msg']}"
            d['msg'] += '\n' + d['eval'] if d['eval'] else ''
            msgs_idxs.append((d['msg'], d['idx']))

        msgs_idxs = sorted(msgs_idxs, key=lambda mi: mi[1])

        msgs = [m for m, i in msgs_idxs]
        self.msgs = msgs
        self.txt = self.SEP.join(msgs)


def get_current_eval_log(log_dir='./logs'):
    f0_txt = None
    f0_v = None

    fns = [fn for fn in os.listdir(log_dir) if re.match(r'rulecheck-eval-v\d+(\.\d+)?\.log$', fn)]
    if len(fns) > 1:
        x = input(
            f'Find multiple log files. If continued, <{fns[0]}> will be read and others may be overwritten ([y]/n)')
        if not (x == '' or x == 'y'):
            exit()

    fn = fns[0]
    f0_v = int(fn[fn.index('-v') + 2:fn.index('.')])
    print(f'Read file: {fn}\n')
    with open(f'{log_dir}/{fn}', 'r', encoding='utf8') as f:
        f0_txt = f.read()

    return f0_v, f0_txt


def update_eval_log(log_dir='./logs', ignore_hash_changes=False):
    """ignore_hash_changes: just copy eval by matched seq_id """
    print('\n=== Process eval log file ===')
    if ignore_hash_changes:
        print('*NOTE: ignore hash changes')

    with open(f'./logs/rulecheck.log', 'r') as f:
        f1_txt = f.read()

    f0_v, f0_txt = get_current_eval_log(log_dir)
    ef0 = EvalLogFile(f0_txt)  # last log file
    ef1 = EvalLogFile(f1_txt)  # current log file

    # ==================== Update
    rct_change = False

    idx0s = [d0['idx'] for _, d0 in ef0.ddict.items()]
    for seq_id1, d1 in ef1.ddict.items():
        if seq_id1 not in ef0.ddict:
            d1['idx'] = 9999
            continue

        # d0, d1 with same seq
        d0 = ef0.ddict[seq_id1]
        d1['idx'] = d0['idx']
        # d1['idx'] = idx0s.index(d0['idx']) + 1  # compact index

        eval0, eval1 = d0['eval'], ''
        if ignore_hash_changes:
            eval1 = eval0
            if d1['rct_id'] != d0['rct_id']:
                rct_change = True
                # print(f"[Ignored] RCT change from {eval0} in current seq {seq_id1})")
        else:
            # d = {'idx': idx, 'seq': seq, 'label': label, 'rct_id': rct_id_, 'eval': eval, 'msg': msg}
            if d1['rct_id'] == d0['rct_id']:
                eval1 = eval0
                if d1['label'] != d0['label']:
                    eval1 += ' (label change)'
                    print(f"Label change in current seq {seq_id1}")
            else:
                rct_change = True
                # ('##correct', '##wrong', '##relabel', '##del')
                if eval0.startswith('##correct'):
                    print(f"!RCT change from {eval0}. in current seq {seq_id1}")
                    eval1 = f"(RCT change from correct) ?{eval0}"
                elif eval0:
                    print(f" RCT change from {eval0}. in current seq {seq_id1}")
                    eval1 = f"(RCT change) ?{eval0}"
                else:
                    print(f" (RCT change in non-eval seq {seq_id1}")
                    # pass

        d1['eval'] = eval1

    # ==================== Print Info (Dont use log)
    if not rct_change:
        print('[No RCT Change]')
    h0 = set(ef0.ddict)
    h1 = set(ef1.ddict)
    h_del = h0 - h1
    h_add = h1 - h0
    print(f'\nLog0 count: {len(h0)}, Log1(current) count: {len(h1)}')
    print(f'Deleted seqs (n={len(h_del)}): {h_del}')
    print(f'Added seqs (n={len(h_add)}): {h_add}')

    # ==================== Write
    with open(f'{log_dir}/rulecheck-eval-v{f0_v + 1}.log', 'w', encoding='utf8') as f:
        SEP = ef0.SEP
        f.write(ef0.msgs[0])
        f.write(SEP)

        ef1.update_eval()
        f.write(ef1.txt)
        f.write(SEP)

        if h_del:
            f.write(f'\n\n##del or not matched')
            f.write(SEP)
            for h in h_del:
                d = ef0.ddict[h]
                f.write(f"[{d['idx']}]{d['msg']}")
                f.write(SEP)

    # ==================== Hash check
    # subprocess.Popen(['powershell', f"md5sum './logs/rulecheck-eval-v*.log'"])
    with open(f'{log_dir}/rulecheck-eval-v{f0_v}.log', 'r', encoding='utf8') as f:
        print(f'\n[Hash] v{f0_v}:', hashlib.md5(f.read().encode('utf8')).hexdigest())
    with open(f'{log_dir}/rulecheck-eval-v{f0_v + 1}.log', 'r', encoding='utf8') as f:
        print(f'[Hash] v{f0_v + 1}:', hashlib.md5(f.read().encode('utf8')).hexdigest())

    # ==================== Measure & Print
    print('\n--')
    df = {'has_app': [], 'rec_pr': [], 'n_props': [], '2_props': [], 'correct': []}
    ids = []
    for h, d in ef1.ddict.items():
        rct = d['msg'].split('\n')[4:-2]
        ids.append(h)
        df['has_app'].append(int(any('-?' in t for t in rct)))
        df['rec_pr'].append(int(any('|--' in t for t in rct)))
        df['n_props'].append(len(rct) - 1)
        df['2_props'].append(int(len(rct) - 1 >= 2))
        df['correct'].append(int(bool(re.match('###?correct', d['eval']))))
    df = pd.DataFrame(df, index=ids)
    # df.to_csv(f'{log_dir}/log-v{f0_v + 1}.csv')

    n = len(df['correct'])
    nc = sum(df['correct'])
    print(f"All={n}, Correct={nc}({nc / n:.4f}), Wrong={n - nc}({(n - nc) / n:.4f})\n")

    print('\t\thas_app\t2_props\trec_pr\tall')
    c_a = sum(df['has_app'] * df['correct']) / sum(df['has_app'])  # rate of correct-app
    c_2 = sum(df['2_props'] * df['correct']) / sum(df['2_props'])
    c_r = sum(df['rec_pr'] * df['correct']) / sum(df['rec_pr'])

    c_na = sum((1 - df['has_app']) * df['correct']) / sum(1 - df['has_app'])
    c_n2 = sum((1 - df['2_props']) * df['correct']) / sum(1 - df['2_props'])
    c_nr = sum((1 - df['rec_pr']) * df['correct']) / sum(1 - df['rec_pr'])

    a2r = df['has_app'] * df['2_props'] * df['rec_pr']
    c_a2r = sum(df['has_app'] * df['2_props'] * df['rec_pr'] * df['correct']) / sum(a2r)
    c_na2r = sum((1 - df['has_app'] * df['2_props'] * df['rec_pr']) * df['correct']) / sum(1 - a2r)

    print(
        f"rate\t{sum(df['has_app']) / n:.4f}\t{sum(df['2_props']) / n:.4f}\t{sum(df['rec_pr']) / n:.4f}\t{sum(a2r) / len(a2r):.4f}")
    print(f"c1  \t{c_a:.4f}\t{c_2:.4f}\t{c_r:.4f}\t{c_a2r:.4f}")
    print(f"c1-non\t{c_na:.4f}\t{c_n2:.4f}\t{c_nr:.4f}\t{c_na2r:.4f}")
    print(f"complex={sum(df['2_props'])}, simple={n - sum(df['2_props'])}\n")

    d_tags = OrderedDict.fromkeys(sorted(TAGS, key=lambda t: 'sobj obj prop cmp Rprop aRprop Robj aRobj'.index(t)), 0)
    df_tag_n = pd.DataFrame(d_tags, index=[0])
    df_tag_n_simp = pd.DataFrame(d_tags, index=[0])
    df_tag_n_comp = pd.DataFrame(d_tags, index=[0])
    for h, d in ef1.ddict.items():
        for t in TAGS:
            nt = d['label'].count(f'/{t}]')
            df_tag_n[t] += nt
            if df.loc[h, '2_props']:
                df_tag_n_comp[t] += nt
            else:
                df_tag_n_simp[t] += nt

    df_tag_n['TOTAL'] = df_tag_n.iloc[0, :].sum()
    df_tag_n_simp['TOTAL'] = df_tag_n_simp.iloc[0, :].sum()
    df_tag_n_comp['TOTAL'] = df_tag_n_comp.iloc[0, :].sum()
    print(f'Tag-all\n{df_tag_n}')
    # print(f'Tag-simple\n{df_tag_n_simp}')
    print(f'Tag-complex\n{df_tag_n_comp}')

    return df, ef1


def _interactive_rct_gen():
    while True:
        try:
            seq_id = input('Input seq/text id:').strip()
            if '#' in seq_id:
                seq_id = seq_id[seq_id.index('#') + 1:]
            if ':' in seq_id:
                seq_id = seq_id[seq_id.index(':') + 1:]
            if ']' in seq_id:
                seq_id = seq_id[seq_id.index(']') + 1:]
            print('-' * 40)

            seqs, labels, dicts_ = init_data_by_json(return_only=True)  # update
            dd = {}
            for d_ in dicts_:
                dd.update({d_['text_id']: {'seq': d_['text'], 'label': d_['label']}})

            seq, label = dd[seq_id]['seq'], dd[seq_id]['label']

            log('-' * 90)
            rct = RCTree(seq, label)
            flag, log_msg = rct.generate()

            log_msg = log_msg[log_msg.index('#'):]  # strip [idx]
            pyperclip.copy(log_msg)
            print('(Copied to clipboard)\n\n')
        except KeyboardInterrupt as ex:
            print('\n*** KeyboardInterrupt, Exit ***')
            break
        except KeyError as ex:
            print('\n*** Invalid seq id, please retry ***')
            continue


def get_args():
    parser = argparse.ArgumentParser('RuleCheckTransform Project')
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-u', '--update_only', action='store_true')
    args_ = parser.parse_args()

    return args_


total_count, complete_count = 0, 0
if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    if args.update_only:
        update_eval_log()
        sys.exit()

    logger = Logger(file_name='rulecheck.log', init_mode='w+')
    log = logger.log
    if args.interactive:
        log('=== Interactive RCT Gen ===')
        _interactive_rct_gen()
        exit()

    log('=== RCTree Generation Start ===')
    log('-' * 90)
    for seq, label in json_data_loader():
        rct = RCTree(seq, label)
        flag, _ = rct.generate()
        total_count += 1
        complete_count += 1 if flag else 0

    log('=== RCTree Generation Finished ===')
    log(f'Complete: {complete_count}/{total_count}={complete_count / total_count:.4f}')
    log(f'Time cost: {get_elapsed_time(start_time)}')

    update_eval_log()
