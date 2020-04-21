'''
@Autor: xujiahuan
@Date: 2020-04-21 13:24:40
@LastEditors: xujiahuan
@LastEditTime: 2020-04-21 14:56:59
'''
import time
from collections import Counter

from models.hmm import HMM
# from models.crf import CRFModel
# from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
from evaluating import Metrics
import joblib
# from sklearn.externals import joblib


def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    """训练并评估hmm模型"""
    # 训练HMM模型
    # print("test_data:", test_data)
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model, "./ckpts/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)
    print(test_tag_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists
