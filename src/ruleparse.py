#!/usr/bin/python3.7
# coding=utf-8

import sys
import argparse
import nltk
import re
import hashlib
import pyperclip
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import List, Tuple
from antlr4parser import *
from utils import *
from data import *

TAGS = {'sobj', 'obj', 'prop', 'cmp', 'Rprop', 'ARprop', 'Robj'}
DEFAULT_TAG_VALUES = {'prop': 'Type', 'cmp': '='}
DEONTIC_WORDS = ('应当', '应该', '应按', '应能', '尚应', '应', '必须', '尽量', '要', '宜', '得', 'shall')
CMP_DICT = OrderedDict([('≤', '小于或?等于|不(大于|高于|多于|超过)|(not be|no) greater than'),
                        ('≥', '大于或?等于|不(小于|低于|少于)|(not be|no) less than'),
                        ('>', '大于|超过|高于|greater than'),
                        ('<', '小于|低于|less than'),
                        ('≠', '不等于|避免|不采用|无法采用|非|(not be|no) equals?'),
                        ('=', '等于|为|采?用|按照?|符合|执行|equals?'),
                        ('has no', '不(有|设置?|具备)|无'),
                        ('has', '(具|含|留)?有|设(置|有)?|(增|铺|搭)设|安装|具备'),
                        ('not in', '不在'),
                        ('in', '在'),
                        ])


def get_cmp_str(cmp_w):
    """ cmp_str: cmp key, cmp_w: cmp word/value """
    if not cmp_w:  # None or ''
        return DEFAULT_TAG_VALUES['cmp']

    for dw in DEONTIC_WORDS:
        cmp_w = cmp_w.replace(dw, '')
    cmp_w = cmp_w.strip()

    # simplify cmp_value, if can
    if cmp_w == '':
        return '='

    for key, pattern in CMP_DICT.items():
        if re.fullmatch(pattern, cmp_w, re.IGNORECASE):
            return key

    return cmp_w


def reverse_cmp(cmp):
    assert cmp in CMP_DICT
    for cmp2 in (('=', '≠'), ('<', '≥'), ('≤', '>'), ('has', 'has no'), ('in', 'not in')):
        if cmp in cmp2:
            return cmp2[1 - cmp2.index(cmp)]


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
        Example1: [(x, obj), (or, OR), (y, obj)] -> [(x|y, obj)]
        Example2: [(x, obj), (and, AND), (y, obj)] -> [(x&y, obj)]
        Example:
            x = LabelWordTags([('x', 'obj'), ('or', 'OR'), ('y', 'obj')])
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

    def remove_o_word(self, word=''):
        """ remove meaningless words, if its tag='O' """
        for i in range(len(self.word_tags) - 1, -1, -1):
            if self.word_tags[i] == (word, 'O'):
                del self.word_tags[i]

    def hashtag(self):
        seq_ = []
        for word, tag in self.word_tags:
            seq_.append(f'{word}#{tag}')

        seq_ = ' '.join(seq_)
        return str_hash(seq_)

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
        rprop = self.to_rcnode(ctx.RPROP())  # AR* or R*
        robj = self.to_rcnode(ctx.ROBJ()) if (ctx.ROBJ() is not None) else None  # 'Robj'

        return cmp, rprop, robj


class RCTree:
    def __init__(self, seq, label_iit, log_func=print):
        """
        :param seq:     sentence, list of char: ['a','b','c',...] -> str: 'abc..'
        :param label_iit:   label_tuple, [(i,j,tag),(i,j,tag),...]
        """
        self.root = RCNode('#', None)
        self.curr_node = self.root  # the node who just add_child
        self.obj_node = None  # shortcut to access obj in the tree

        self.seq = seq
        self.seq_id = str_hash(self.seq)
        self.label_iit = label_iit  # [(i,j,tag),(i,j,tag),...]
        # self.strip_seq_label()
        self.full_label = self.get_full_label()  # flabel_wt: LabelWordTags
        self.slabel = str(self.full_label)

        self.parse_complete = False
        self.error_msg = ''
        self.log_func = log_func

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
        # ======================================== Bool merge (union)
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
            [self.full_label.bool_merge(lwts) for lwts in result3]
            result2 = self.regex_parse(f'X: {{<{tag1}><OR><{tag1}><OR><{tag1}>}}')
            [self.full_label.bool_merge(lwts) for lwts in result2]
            result1 = self.regex_parse(f'X: {{<{tag1}><OR><{tag1}>}}')
            [self.full_label.bool_merge(lwts) for lwts in result1]

        for i, (w, t) in enumerate(self.full_label):
            if t == 'OR':
                self.full_label.rename(i, tag='O')

        # ======================================== Eliminate meaningless words and duplicated O-tag
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

        # ======================================== Sub-seq switch
        # 1) A 应符合下列规定： B 时 C -> B A C
        def _sub_seq_switch(i, j):
            if i < j < len(self.full_label) - 1:
                print(f'[DEBUG] sub-seq switch at {self.full_label[i:j]}')
                ar_wts = self.full_label.word_tags[i:j]  # AR
                self.full_label.remove(list(range(i, j)))
                self.full_label.word_tags += ar_wts  # 可以考虑对ar_wts单独生成一个sub-tree

        for i, (wi, ti) in enumerate(self.full_label):
            if ti == 'O':
                js = [j for j, (wj, tj) in enumerate(self.full_label[i + 1:]) if tj == 'O' and '时' in wj]
                if js and (re.search(r'^(应符合|不应.于)下列规定：', wi) or wi.strip(' ,，') in ('在', '当')):
                    j = i + 1 + js[0]
                    has_arprop = any(t == 'ARprop' for w, t in self.full_label[i + 1:j])
                    has_no_obj = all(t != 'obj' for w, t in self.full_label[i + 1:j])
                    if has_arprop and has_no_obj:
                        j = i + 1 + js[0]
                        _sub_seq_switch(i, j)
                        break
                if re.search(r'^(应符合|不应.于)下列规定：', wi):
                    lwtss = self.regex_parse('X: {<O><.*>*<ARprop><O>}', self.full_label[i:])  # greedy
                    if len(lwtss) == 1:
                        lwts = lwtss[0]
                        j = i + len(lwts) - 1  # tag-j is O
                        _sub_seq_switch(i, j)
                        break

        # 2) 当A被B C时 -> 当B把A C时
        for i, lwts in self.regex_parse('X: {<Robj>?<A?Rprop><O><obj><prop>}', return_idx=True):
            if i >= 1 and '当' in self.full_label[i - 1][0] and self.full_label[i + len(lwts) - 3][0] == '被':
                self.full_label.rename(i + len(lwts) - 3, '把')
                _switch_full_label_s(i, i + len(lwts) - 2)
                _switch_full_label_s(i + 1, i + len(lwts) - 2)

        # ======================================== Has prop
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
        for i, lwts in self.regex_parse('X: {<cmp><cmp><Robj>?<A?Rprop><prop>}', return_idx=True):
            if _is_cmp_has(*lwts[0]):
                _switch_full_label_s(i + 1, i + len(lwts) - 1)
                self.full_label.insert(i + 1, (', ', 'O'))  # avoid p-r swtich here
                self.full_label.insert(i + 1, (lwts[-1][0], lwts[-2][1]))

        # 3.0) has c? R1 p1 R2 -> has p1 c? R1 R2 (p-r switch for 3.1)
        for i, lwts in self.regex_parse('X: {<cmp><cmp>?<Rprop><prop><Rprop>}', return_idx=True):
            if _is_cmp_has(*lwts[0]):
                _switch_full_label_s(i + 1, i + len(lwts) - 2)

        # 3.1) has/eq p1 c? R p2 -> [has Rp2,] p2 p1 c? R
        for i, lwts in self.regex_parse('X: {<cmp><prop><cmp>?<Robj>?<A?Rprop><R?prop>}', return_idx=True):
            # while lwts := regex()
            c_str = get_cmp_str(lwts[0][0])
            if c_str in ('has', '='):
                self.full_label.rename(i + len(lwts) - 1, tag='prop')
                _switch_full_label_s(i + 1, i + len(lwts) - 1)
                if c_str == 'has':
                    self.full_label.insert(i + 1, (', ', 'O'))  # avoid p-r swtich here
                    self.full_label.insert(i + 1, (lwts[-1][0], lwts[-2][1]))  # can be AR OR R
                else:
                    self.full_label.remove(i)
                print(f'[DEBUG] (has prop) p-r switch and insert Rprop in {self.full_label[i:i + 3]}')

        # ======================================== Prop-Rprop (p-r) switch
        for w1 in ('作', '作为', '进行', '处'):
            self.full_label.remove_o_word(w1)

        # ========== (c ro r) order
        # 0.1) ro c r -> c ro r
        for i, lwts in self.regex_parse('X: {<Robj><cmp><A?Rprop>}', return_idx=True):
            _switch_full_label_s(i, i + 1)

        # 0.2) c r ro -> c ro r
        for i, lwts in self.regex_parse('X: {<cmp><A?Rprop><Robj>}', return_idx=True):
            if i + 2 == len(self.full_label) - 1 or not self.full_label[i + 3][1].endswith('Rprop'):
                _switch_full_label_s(i + 1, i + 2)

        # 0.3) p c ro O -> p c ro p O
        for i, lwts in self.regex_parse('X: {<prop><cmp><Robj>}', return_idx=True):
            if not any(t == 'Rprop' for w, t in self.full_label[i + 2:i + 5]):
                self.full_label.insert(i + 3, (lwts[0][0], 'Rprop'))

        # ========== C r p
        # 1.1) C r p (Note: 1.x cannot use self.regex_parse because it may return over-lapping word-tags)
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
        for i, lwts in self.regex_parse('X: {^<O|obj>*<Robj>?<A?Rprop><prop><O>}', return_idx=True):
            j = int(lwts.contains_tags('Robj'))  # 0 or 1
            _switch_full_label_s(i + len(lwts) - 3 - j, i + len(lwts) - 2)

        # TODO p1 p2 p3 xxx R -> match p2-R not p1-R
        #  p1 C R 的 p2 -> p2 p1 C R
        # ### px p r p
        # for i, ((w0, t0), (w1, t1), (w2, t2), (w3, t3)) in _enum_full_label(4, reverse=True):
        #     if t0 == 'prop' and t1 == 'cmp' and t2.endswith('Rprop') and t3 == 'prop':
        #         _switch_full_label_s(i, i + 3)

        # ======================================== Match prop
        while self.regex_parse('X: {<prop><cmp><Robj>?<Rprop><O><cmp><Rprop>}', return_idx=True):
            i, lwts = self.regex_parse('X: {<prop><cmp><Robj>?<Rprop><O><cmp><Rprop>}', return_idx=True)[0]
            n = len(lwts)
            if lwts[n - 3][0].strip('，') in {'并', '且', '并且', '即'}:
                self.full_label.insert(i + n - 2, lwts[0])
                print(f"[DEBUG] pre match p-r tag for in {lwts}")
            else:
                break

        # ======================================== Add order indicator in self full_label and slabel ([word-i/tag])
        self.slabel_2 = str(self.full_label)  # slabel after preprocess2
        for i in range(len(self.full_label)):
            w, t = self.full_label[i]
            if t not in ('sobj', 'obj', 'O'):
                self.full_label.rename(i, f'{w}-{i}', t)
        self.slabel_2i = str(self.full_label)  # slabel_2 with order indicator

        print(f'[Debug] LabelX:\t{self.full_label}')

    def post_process(self):
        self.obj_node: RCNode
        if not self.obj_node:
            return

        # ========== Sort props
        def _n_str(prop: RCNode):
            # if '|' in s: s = s.split('|')[0]
            return prop.__str__(optimize=False)[:-1]

        def _sort_child_nodes(node: RCNode):
            for cn in node.child_nodes:
                _sort_child_nodes(cn)

            idx_props = []
            for i, p in enumerate(node.child_nodes):
                ind = self.slabel_2i.find(_n_str(p))
                if p.req:
                    ind = self.slabel_2i.find(_n_str(p.req[1]))
                assert ind >= 0
                idx_props.append((not p.has_app_seq(), ind, i, p))
            idx_props.sort()
            node.child_nodes = [p for *_, p in idx_props]

        _sort_child_nodes(self.obj_node)

        # ========== Add bool indicator (OR*)
        for i, p in enumerate(self.obj_node.child_nodes):
            if i == 0:
                continue
            p1 = self.obj_node.child_nodes[i - 1]
            if p.req and p1.req and (p.is_app_req() == p1.is_app_req()):
                p1i = max(self.slabel_2i.find(_n_str(n)) for n in (p1, p1.req[1], p1.req[2]) if n)
                pis = [self.slabel_2i.find(_n_str(n)) for n in (p, p.req[1], p.req[2]) if n]
                if not pis or all(i < 0 for i in pis):
                    continue
                pi = min(i for i in pis if i >= 0)
                if 0 <= p1i < pi:
                    s = self.slabel_2i[p1i:pi]
                    s = s[s.find(']') + 1:]
                    if s and s[0] == '或':
                        p.or_combine = True
                        print(f'[DEBUG] OR combine: {p1}, {p}')

        # ========== Remove order indicator in nodes
        def _rm_oder_indicator(nodes):
            for p in nodes:
                _rm_oder_indicator(p.child_nodes)
                if p:
                    p.word = p.word[:p.word.rindex('-')]
                if p.req:
                    if p.req[0]:
                        p.req[0].word = p.req[0].word[:p.req[0].word.rindex('-')]
                    if p.req[1]:
                        p.req[1].word = p.req[1].word[:p.req[1].word.rindex('-')]
                    if p.req[2]:
                        p.req[2].word = p.req[2].word[:p.req[2].word.rindex('-')]

        _rm_oder_indicator(self.obj_node.child_nodes)

        # ==========
        props: List[RCNode] = self.obj_node.child_nodes
        full_label_bak = self.full_label.copy()
        self.full_label = self.get_full_label()
        self.pre_process1()  # update self.full_label

        # ========== Default prop, p-r match, match last prop
        def _is_union_word(w_: str):
            for dw in DEONTIC_WORDS:
                w_ = w_.replace(dw, '')
            # do not use ('且', '并'): [熔点/prop]_[不小于/cmp]_[1000℃/ARprop]_且_[无/cmp]_[绝热层/ARprop]_的...
            patterns = {'，并$', '，且$', '，并且$', '，即$', '^或'}
            return any(re.search(p, w_) for p in patterns)

        for i in range(1, len(props)):  # use ascending order
            # p R 并/且|其 R
            pi = props[i]
            pi1 = props[i - 1]
            if pi is None or pi1 is None:
                continue
            is_prr = bool(pi1.word and pi1.req and not pi.word and pi.req)
            is_pxp = bool(pi1.word and pi1.child_nodes and pi.word)
            is_same_req = pi.is_app_req() == pi1.is_app_req()
            if is_prr and is_same_req:
                li1 = self.full_label.index((pi1.word, pi1.tag))  # last prop
                if li1 < 0 or li1 > len(self.full_label) - 3:
                    continue
                req = [(r.word, r.tag) for r in pi.req if r]
                full_label_1 = self.full_label[li1 + 1:]
                # li = the first valid index of req
                if len(req) <= 2:
                    li = full_label_1.index(req)
                    if li < 0:
                        li = full_label_1.index(req[::-1])
                else:
                    lis = [li for li in (full_label_1.index(r_) for r_ in (req, [req[0], req[2], req[1]], req[:2]))
                           if li >= 0]
                    li = lis[0] if lis else -1
                li += li1 + 1
                is_next = (0 <= li1 < li < len(self.full_label)) and all(
                    t != 'prop' for w, t in self.full_label[li1 + 1:li])
                if is_next and _is_union_word(self.full_label[li - 1][0]):
                    pi.word = pi1.word
                    if not pi.req[0]:  # cmp
                        pi.req[0].word = pi1.req[0].word
                    print(f"[DEBUG] match p-r tag for {str(pi1)} in {self.full_label[li1:li + 1]}")
            elif is_pxp:
                if pi.word == '其' and pi.child_nodes and is_same_req:
                    print(f'[DEBUG] match last prop-1 {pi}')
                    pi.word = pi1.word
                    # pi1.child_nodes += pi.child_nodes # 2 propx
                    # props[i] = None
                elif all(pi.word == cn.word for cn in pi1.child_nodes):
                    print(f'[DEBUG] match last prop-2 {pi}')
                    pi1.child_nodes.append(pi)
                    props[i] = None

        for i in range(len(props) - 1, -1, -1):
            if props[i] is None:
                del props[i]

        # ========== Remove duplicated nesting prop-propx, match prop
        def _rm_duplicated_child_prop(props_):
            for i_, prop_ in enumerate(props_):
                _rm_duplicated_child_prop(prop_.child_nodes)
                if prop_.n_child() == 1 and prop_.word == prop_.child_nodes[0].word and prop_.tag == \
                        prop_.child_nodes[0].tag:
                    props_[i_] = prop_.child_nodes[0]

        _rm_duplicated_child_prop(props)

        # ========== Match ARprop-obj (anchor)
        if ', ' in self.obj_node.word:
            words = self.obj_node.word.split(', ')
            anchored = False
            slabel = str(self.full_label)
            iobjs = [slabel.index(f'[{w}/obj]') for w in words]
            for prop in props:
                if prop.is_app_req() and not isinstance(prop.word, list):
                    wp = prop.word if prop.word is not None else ''
                    ip = re.search(f'{wp}.*?{prop.req[1].word}/ARprop', slabel).span()[1]
                    for k, io in enumerate(iobjs):
                        if ip < io:
                            prop.anchor = f'&{k}' if ('时，' in slabel[ip:io]) else f'&{k + 1}'
                            anchored = True
                            break
                    if not prop.anchor:
                        prop.anchor = f'&{len(iobjs)}'
                        anchored = True
            if anchored:
                for i, w in enumerate(words):
                    words[i] += f'&{i + 1}'
                    self.obj_node.word = ', '.join(words)

        self.full_label = full_label_bak

    def regex_parse(self, grammar, full_label=None, return_idx=False):
        """
        :param grammar: e.g., 'P: {<prop><propx><cmp>?<A?Rpropx>}'
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

    def cfg_parse(self):
        def _antlr4_parse():
            input_stream = InputStream(str(self.full_label))
            lexer = RuleCheckTreeLexer(input_stream)
            tokens = CommonTokenStream(lexer)
            parser = RuleCheckTreeParser(tokens)
            parser._listeners = [RCTreeErrorListener()]
            # parser.addErrorListener(RCTreeErrorListener())
            tree = parser.rctree()
            # print(tree.toStringTree(recog=parser))
            return tree

        self.full_label.remove_tag('O')
        tree = _antlr4_parse()
        if not tree:
            return
        visitor = RCTreeVisitor()
        visitor.visit(tree)
        self.full_label.remove(visitor.all_wts)
        for p_node in visitor.sprop_nodes:
            self.obj_node.add_child(p_node)

    def parse(self):
        """ Parse RCTree based on CFG and regex """
        self.pre_process1()

        def _classes_same(ws: list):
            """two word is in the same class, e.g., 金属管道/铸铁管道 都属于管道
            it should be the work of ontology, and now this is a rough implementation"""
            if len(ws) <= 1:
                return True
            return all(ws[i][-2:] == ws[i + 1][-2:] for i in range(len(ws) - 1))

        if 'sobj' in self.full_label.tags:
            i_sobjs, w_sobjs = self.full_label.tag_idxs_words('sobj')
            if len(i_sobjs) == 1:
                self.add_curr_child(RCNode(w_sobjs[0], 'sobj'))
            else:
                while w_sobjs:
                    for i in range(len(w_sobjs), 0, -1):
                        if _classes_same(w_sobjs[:i]):
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

        try:
            self.cfg_parse()
        except ParserError as ex:
            self.error_msg = str(ex)

        self.post_process()

        self.parse_complete = self.full_label.contains_tags(('O', 'obj'), only=True)

    def log_msg(self, idx=0):
        self.log_func('-' * 90)
        self.log_func(f'[{idx}]#{self.seq_id}')
        self.log_func(f'Seq:\t{self.seq}')
        self.log_func(f'Label:\t{self.slabel}')

        if self.error_msg:
            self.log_func(self.error_msg)

        self.log_func(f"RCTree:\t#{self.hashtag()}\n{self}")
        self.log_func('Parsing complete' if self.parse_complete else f'Parsing incomplete: {self.full_label}')

    def hashtag(self):
        return str_hash(self.__str__(indent='\t\t'))

    def __str__(self, indent='\t\t'):
        """
        [sobj1]-[obj1]
        |-?[prop1] = [ARprop1]-[Robj1]
        |-[prop2] = [Rprop1]-[Robj2]
        |-[prop3]
        |--[prop4] < [Rprop2]
        """

        # ========== Obj (consider first child only)
        assert self.root.n_child() == 1
        node = self.root.child_nodes[0]
        tree = indent + str(node)  # str(self.root)
        while node is not self.obj_node:
            assert node.n_child() == 1
            node = node.child_nodes[0]
            tree += f"-{str(node)}"

        assert self.obj_node.req is None  # ignore obj's req now
        if not self.obj_node.child_nodes:
            return tree

        obj_tree = self.obj_node.tree_str()
        obj_lines = obj_tree.split('\n')[1:]

        obj_lines = [indent + l for l in obj_lines]
        obj_tree = '\n'.join(obj_lines)

        return tree + '\n' + obj_tree


class RCNode:
    """The basic element in RCTree, it can be an object/property, or a union/intersection of them """

    def __init__(self, word, tag):
        if isinstance(word, list) or isinstance(word, tuple):
            word = [w for i, w in enumerate(word) if w not in word[:i]]  # remove duplications
            for i in range(len(word) - 1, 0, -1):  # remove duplications
                if word[i - 1].endswith(word[i]) and '|' not in word[i - 1]:
                    del word[i]
            if isinstance(word, list) or isinstance(word, tuple):
                word = ', '.join(word)

        self.word = word
        self.tag = tag
        self.onto_name = None
        self.onto_type = None

        self.child_nodes = []
        self.req = None  # a tuple of (cmp_node, req_node, sreq_node=None)

        self.anchor = ''  # anchor to a specific obj, when there are multiple objs
        self.or_combine = False  # bool condition, default (False) is AND

    def is_app_req(self):
        if self.req:
            return self.req[1].tag[0] == 'A'
        return None

    def has_app_seq(self):
        if self.req:
            return self.req[1].tag[0] == 'A'
        elif self.child_nodes:
            return any(cn.has_app_seq() for cn in self.child_nodes)
        else:
            return False

    def add_child(self, node):
        self.child_nodes.append(node)

    def n_child(self):
        return len(self.child_nodes)

    def set_req(self, req: tuple):
        assert self.req is None, 'Each RCNode has at most one req'
        assert len(req) == 3
        assert 'Rprop' in req[1].tag  # '[A]Rprop'

        self.req = req

    def set_req_by_wts(self, wts):
        """ wts: list/tuple of word-tag (<O> tags will be filtered)"""
        if self.tag == 'obj':
            ct, rt, srt = 'cmp', 'Robj', None  # may be 'Rsobj' in the future
        elif self.tag == 'prop':
            ct, rt, srt = 'cmp', 'Rprop', 'Robj'
        else:
            raise AssertionError('add_req_by_tuple() should be used for only /obj, /prop')
        if any(('AR' in wt[1] for wt in wts)):
            rt = 'A' + rt

        # Double check
        for w, t in wts:
            assert t in (ct, rt, srt, 'O') or (w, t) == (self.word, self.tag)

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

    def tree_str(self, indent='-', optimize=True, show_tag=False):
        t_str = self.__str__(optimize, show_tag, True)

        if self.child_nodes:
            # cns = sorted(self.child_nodes, key=lambda cn: not cn.has_app_seq()) # have been sorted by post_process
            sep_and = f'\n|{indent}'
            sep_or = sep_and[:-1] + '+'
            cn_t_strs = [(sep_or if cn.or_combine else sep_and, cn.tree_str(indent + '-', optimize))
                         for cn in self.child_nodes]
            t_str += ''.join(s for ss in cn_t_strs for s in ss)

        return t_str

    def hashtag(self):
        all_str = self.tree_str(optimize=False)
        return str_hash(all_str)

    def __str__(self, optimize=True, show_tag=False, show_req=False):
        word = self.word

        if optimize:
            if not word:
                word = DEFAULT_TAG_VALUES[self.tag] if self.tag in DEFAULT_TAG_VALUES else '?'
                if self.tag == 'prop' and self.req and str(self.req[0]).startswith('has'):
                    word = 'Props'
            if self.tag == 'cmp':
                return get_cmp_str(word)
            # TODO default_cmp_value 高于: 位置 大于, etc

        t = f'/{self.tag}' if show_tag else ''
        o = f':{self.onto_name}' if self.onto_name else ''
        str_ = f'[{word}{self.anchor}{o}{t}]'

        if show_req:
            if self.req:
                # req_str_: cmp, Rprop [,Robj]
                req_str_ = f"{self.req[0].__str__(optimize)} {self.req[1].__str__(optimize)}"
                if self.req[2] is not None:
                    req_str_ += f"-{self.req[2].__str__(optimize)}"
                str_ = f"{str_} {req_str_}"
                # ? means if
                prefix = '?' if self.is_app_req() else ''
                str_ = prefix + str_
            else:
                pass

        return str_

    def __bool__(self):
        return bool(self.word)


class RevitRuleGenerator:
    Class_Param_Names = (('OST_Doors', 'Doors?$', '门$', {}),
                         ('OST_Stairs', 'Stairs?$', '楼梯$', {'^宽度$': '最小梯段宽度', '^Width$': 'Minimum Run Width'}),
                         ('OST_Windows', 'Windows?$', '窗$|^窗', {}),
                         ('OST_Walls', 'Walls?$', '墙$', {'Type': 'Structural Material'}),  # 结构材质
                         ('OST_StairsRailing', 'Railings?$', '栏杆$', {'高度': '栏杆扶手高度'}),
                         ('OST_Floors', 'Floors?$', '楼板$', {'Thickness$': 'Default Thickness', '厚度': '默认的厚度'})
                         )
    Param_Names = {'混凝土$': 'Concrete', '^热阻$': '热阻(R)', '^thermal resistance$': 'Thermal Resistance (R)'}
    Cmp_Condition = {'=': 'Equal', '<': 'LessThan', '>': 'GreaterThan', '≤': 'LessOrEqual', '≥': 'GreaterOrEqual',
                     '≠': 'NotEqual', 'has': 'Contains', 'has no': 'DoesNotContain'}
    Root = ET.Element('MCSettings', attrib={'AllowRequired': 'False', 'Name': 'CheckSet-Zyc', 'Author': 'Zyc'})
    Heading = ET.SubElement(Root, 'Heading', attrib={'HeadingText': 'Test', 'IsChecked': 'True'})
    Section = ET.SubElement(Heading, 'Section', attrib={'SectionName': 'Test', 'IsChecked': 'True'})

    def __init__(self, rct: RCTree):
        self.rct = rct
        self.obj = self.rct.obj_node
        self.class_name = None
        self.param_names = self.Param_Names.copy()
        for cn, w_en, w_cn, pns in self.Class_Param_Names:
            if re.search(w_en, self.obj.word, re.IGNORECASE) or re.search(w_cn, self.obj.word):
                self.class_name = cn
                self.param_names.update(pns)
                break
        if self.class_name is None:
            raise RuntimeError('Class name not found:', self.obj.word)

    def get_category_filter(self, op='And'):
        """ <Filter Operator="And" Category="Category" Property="OST_Doors"
            Condition="Included" Value="True" CaseInsensitive="False" Unit="None" UnitClass="None" FieldTitle=""
            UserDefined="False" Validation="None" /> """

        attrib = {'Operator': op, 'Category': "Category", 'Property': self.class_name,
                  'Condition': "Included", 'Value': "True"}
        return attrib

    def to_param_name(self, pw):
        for pw1, pn in self.param_names.items():
            if re.search(pw1, pw, re.IGNORECASE):
                return pn

        if re.search('[a-zA-Z]', pw):
            pw = pw.title()

        return pw

    def get_param_filter(self, node, op='And'):
        """ <Filter Operator="And" Category="Parameter" Property="标高"
            Condition="WildCard" Value="标高3" CaseInsensitive="False" Unit="None" UnitClass="None" FieldTitle=""
            UserDefined="False" Validation="None" /> """

        pv = node.word
        if pv is None:
            pv = str(node).strip('[]')
        pv = self.to_param_name(pv)  # replace prop name

        cmp, rprop, robj = node.req
        if robj is not None:
            raise NotImplementedError('Robj is not support now')

        unit_class = "None"
        rv = self.to_param_name(rprop.word)
        if ' ' in rv and re.search('[0-9]', rv):
            i = rv.rindex(' ')
            rv, u = rv[:i], rv[i + 1:].lower()
            if u == 'm':
                rv = str(float(rv) * 1000 / 304.8)  # in revit, length unit always uses foot (ft)
                unit_class = "Length"
            elif u == 'mm':
                rv = str(float(rv) / 304.8)
                unit_class = "Length"

        cmp_str = str(cmp)
        if not node.is_app_req():
            cmp_str = reverse_cmp(str(cmp))

        cond_str = self.Cmp_Condition[cmp_str]
        if not re.search('[0-9]', rv) and 'Equal' in cond_str:
            cond_str = 'WildCard' if cond_str == 'Equal' else 'WildCardNoMatch'

        attrib = {'Operator': op, 'Category': "Parameter", 'Property': pv, 'Condition': cond_str,
                  'Value': rv, 'Unit': "Default", 'UnitClass': unit_class}

        if pv == 'Type':  # patching
            attrib['Category'] = 'Type'
            attrib['Property'] = 'Name'
        return attrib

    def get_is_elem_filter(self, op='And'):
        """ <Filter Operator="And" Category="TypeOrInstance"
        Property="Is Element Type" Condition="Equal" Value="False" CaseInsensitive="False" Unit="None"
        UnitClass="None" FieldTitle="" UserDefined="False" Validation="None" /> """

        attrib = {'Operator': op, 'Category': "TypeOrInstance", 'Property': "Is Element Type", 'Condition': "Equal",
                  'Value': "0", 'Unit': "Default", 'UnitClass': "None"}
        return attrib

    def generate(self, write_xml=False):
        if not self.obj:
            return
        if self.rct.root.child_nodes[0] is not self.obj:
            print('[Warning] sobj is ignored now') # raise

        # if a then b = a -> b = !a or b, fail: !(!a or b) = a and !b
        check1 = ET.SubElement(RevitRuleGenerator.Section, 'Check', {'CheckName': self.rct.seq,
                                                                     'ResultCondition': 'FailMatchingElements',
                                                                     'IsChecked': 'True'})
        ET.SubElement(check1, 'Filter', self.get_is_elem_filter())
        ET.SubElement(check1, 'Filter', self.get_category_filter())
        for prop in self.obj.child_nodes:
            if prop.child_nodes:
                raise NotImplementedError('propx is not support now')
            ET.SubElement(check1, 'Filter', self.get_param_filter(prop))

        if write_xml:
            ET.ElementTree(RevitRuleGenerator.Root).write('./logs/checkset.xml', encoding='utf-8', xml_declaration=True)


def model_data_loader():
    seqs_raw, labels_raw, _ = init_data_by_json()
    train_data_loader, val_data_loader, corpus = get_data_loader('../data/xiaofang', batch_size=1,
                                                                 check_stratify=False, shuffle_train=False)

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
            Parsing complete.
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
            if lines[i].startswith('Parsing complete') or lines[i].startswith('Parsing incomplete'):
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
    fns = [fn for fn in os.listdir(log_dir) if re.match(r'ruleparse-eval-v\d+(\.\d+)?\.log$', fn)]
    if len(fns) > 1:
        fns.sort()
        x = input(f"Proceed? <{fns[0]}> will be used and <{', '.join(fns[1:])}> may be overwritten (y/[n])")
        if x.lower() != 'y':
            exit()

    if not fns:
        fn = 'ruleparse-eval.log'
        f0_v = 1
        assert os.path.exists(f'{log_dir}/{fn}')
        shutil.copy(f'{log_dir}/{fn}', f'{log_dir}/{fn[:-4]}-v1.log')
    else:
        fn = fns[0]
        f0_v = int(fn[fn.index('-v') + 2:fn.index('.')])

    print(f'Read file: {fn}\n')
    with open(f'{log_dir}/{fn}', 'r', encoding='utf8') as f:
        f0_txt = f.read()

    return f0_v, f0_txt


def update_eval_log(log_dir='./logs', ignore_rct_hash=False):
    """ignore_rct_hash: just copy eval by matched seq_id """
    print('\n=== Process eval log file ===')
    if ignore_rct_hash:
        print('*NOTE: ignore rct hash changes')

    with open(f'./logs/ruleparse.log', 'r') as f:
        f1_txt = f.read()

    f0_v, f0_txt = get_current_eval_log(log_dir)
    ef0 = EvalLogFile(f0_txt)  # last log file
    ef1 = EvalLogFile(f1_txt)  # current log file

    # ==================== Update
    rct_change = False

    idx0s = [d0['idx'] for _, d0 in ef0.ddict.items()]
    # assert len(ef0.ddict) == len(ef1.ddict)  # test
    for i, (seq_id1, d1) in enumerate(ef1.ddict.items()):
        # get d0, d1 with same seq
        if seq_id1 not in ef0.ddict:
            d1['idx'] = 9999
            continue
        d0 = ef0.ddict[seq_id1]
        # d0 = list(ef0.ddict.values())[i]  # test

        d1['idx'] = d0['idx']
        d1['idx'] = idx0s.index(d0['idx']) + 1  # compact index

        eval0, eval1 = d0['eval'], ''
        if ignore_rct_hash:
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
    with open(f'{log_dir}/ruleparse-eval-v{f0_v + 1}.log', 'w', encoding='utf8') as f:
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
    with open(f'{log_dir}/ruleparse-eval-v{f0_v}.log', 'r', encoding='utf8') as f:
        sha1_f0 = hashlib.sha1(f.read().encode('utf8')).hexdigest()
        print(f'\nv{f0_v} sha1 hash:', sha1_f0)
    with open(f'{log_dir}/ruleparse-eval-v{f0_v + 1}.log', 'r', encoding='utf8') as f:
        sha1_f1 = hashlib.sha1(f.read().encode('utf8')).hexdigest()
        print(f'v{f0_v + 1} sha1 hash:', sha1_f1)

    # ==================== Measure & Print
    print('\n-Stat')
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
    print(f"All={n}, correct={nc}({nc / n:.4f}), wrong={n - nc}({(n - nc) / n:.4f})")

    # print('\t\thas_app\t2_props\trec_pr\tall')
    # c_a = sum(df['has_app'] * df['correct']) / sum(df['has_app'])  # rate of correct-app
    # c_2 = sum(df['2_props'] * df['correct']) / sum(df['2_props'])
    # c_r = sum(df['rec_pr'] * df['correct']) / sum(df['rec_pr'])
    #
    # c_na = sum((1 - df['has_app']) * df['correct']) / sum(1 - df['has_app'])
    # c_n2 = sum((1 - df['2_props']) * df['correct']) / sum(1 - df['2_props'])
    # c_nr = sum((1 - df['rec_pr']) * df['correct']) / sum(1 - df['rec_pr'])
    #
    # a2r = df['has_app'] * df['2_props'] * df['rec_pr']
    # c_a2r = sum(df['has_app'] * df['2_props'] * df['rec_pr'] * df['correct']) / sum(a2r)
    # c_na2r = sum((1 - df['has_app'] * df['2_props'] * df['rec_pr']) * df['correct']) / sum(1 - a2r)
    #
    # print(
    #     f"rate\t{sum(df['has_app']) / n:.4f}\t{sum(df['2_props']) / n:.4f}\t{sum(df['rec_pr']) / n:.4f}\t{sum(a2r) / len(a2r):.4f}")
    # print(f"c1  \t{c_a:.4f}\t{c_2:.4f}\t{c_r:.4f}\t{c_a2r:.4f}")
    # print(f"c1-non\t{c_na:.4f}\t{c_n2:.4f}\t{c_nr:.4f}\t{c_na2r:.4f}")

    print(f"Complex={sum(df['2_props'])}")
    print(f"Simple={n - sum(df['2_props'])}")

    if sha1_f0 != sha1_f1:  # show_df_tag
        d_tags = OrderedDict.fromkeys(sorted(TAGS, key=lambda t: 'sobj prop cmp Rprop ARprop Robj'.index(t)), 0)
        df_tag = pd.DataFrame(d_tags, index=['Simple', 'Complex', 'All'])
        for h, d in ef1.ddict.items():
            for t in TAGS:
                nt = d['label'].count(f'/{t}]')
                df_tag[t]['All'] += nt
                if df.loc[h, '2_props']:
                    df_tag[t]['Complex'] += nt
                else:
                    df_tag[t]['Simple'] += nt
        df_tag['TOTAL'] = 0
        for ind in df_tag.index:
            df_tag['TOTAL'][ind] = df_tag.loc[ind].sum()
        print('')
        print(df_tag)

    return df, ef1


def interactive_rct_parse():
    while True:
        try:
            seq = input('Input a sentence or its id:').strip('# ')
            if '[' in seq:
                seq, label = slabel_to_seq_label_iit(seq)
            elif len(seq) == 7:
                seq_id = seq
                dd = {}
                *_, dicts = init_data_by_json(early_return=True)  # update
                for d in dicts:
                    dd.update({d['text_id']: {'seq': d['text'], 'label': d['label']}})
                seq, label = dd[seq_id]['seq'], dd[seq_id]['label']
            else:
                raise KeyError

            rct = RCTree(seq, label)
            rct.parse()
            rct.log_msg()
            log('-' * 90)

            if args.gen_rule:
                rg = RevitRuleGenerator(rct)
                rg.generate(write_xml=True)
        except KeyError as ex:
            print('Invalid input')
            continue
        except (KeyboardInterrupt, EOFError) as ex:
            print(f'\n[Exit] {ex}')
            break


def get_args():
    parser = argparse.ArgumentParser('ARC Rule Parser')
    parser.add_argument('-d', '--dataset_name', type=str, default='json', help='dataset name, json or text')
    parser.add_argument('-g', '--gen_rule', action='store_true', help='generate rule')
    parser.add_argument('-i', '--interactive', action='store_true', help='interactive rct parse')
    parser.add_argument('-U', '--no_update_eval', action='store_true', help='do not update eval log file')
    args_ = parser.parse_args()

    if args_.dataset_name != 'json':
        args_.no_update_eval = True

    return args_


if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    logger = Logger(file_name='ruleparse.log', init_mode='w+')
    log = logger.log

    if args.interactive:
        log('=== Interactive RCTree Parsing (Ctrl-C to exit) ===')
        interactive_rct_parse()
        exit()

    n_parse, n_complete = 0, 0
    log('=== RCTree Parsing Start ===')
    for seq, label in seq_data_loader(args.dataset_name):
        rct = RCTree(seq, label, log)
        rct.parse()
        rct.log_msg(n_parse + 1)
        n_parse += 1
        n_complete += 1 if rct.parse_complete else 0
        if args.gen_rule:
            rg = RevitRuleGenerator(rct)
            rg.generate()
    log('-' * 90)
    log(f'\nComplete: {n_complete}/{n_parse}={n_complete / n_parse:.4f}')
    log(f'Time cost: {get_elapsed_time(start_time)}')

    if not args.no_update_eval:
        update_eval_log()
    if args.gen_rule:
        ET.ElementTree(RevitRuleGenerator.Root).write('./logs/checkset.xml', encoding='utf-8', xml_declaration=True)
