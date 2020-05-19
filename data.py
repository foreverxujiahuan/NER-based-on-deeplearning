'''
@Autor: xujiahuan
@Date: 2020-04-21 12:47:21
@LastEditors: xujiahuan
@LastEditTime: 2020-05-19 19:42:04
'''
from os.path import join
from codecs import open
from collections import Counter


def build_corpus(path, make_vocab=True):
    """读取数据"""
    word_lists = []
    tag_lists = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        word_list = []
        tag_list = []
        for line in f:
            # print(line, len(line))
            if line != '\n' and len(line) > 4:
                # print(">5")
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                # print(line, len(line))
                if len(word_list) != 0 and len(tag_list) != 0:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        # print(temp, len(word_lists))
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
