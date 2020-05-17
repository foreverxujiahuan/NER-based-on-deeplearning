'''
@Autor: xujiahuan
@Date: 2020-05-06 11:00:17
@LastEditors: xujiahuan
@LastEditTime: 2020-05-17 15:26:13
'''
from sklearn.externals import joblib
from data import build_corpus
from utils import extend_maps

train_path = 'data/train.txt'
# dev_path = 'data/dev.txt'
# test_path = 'data/test.txt'
train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus(train_path)
# dev_word_lists, dev_tag_lists = build_corpus(dev_path, make_vocab=False)
# test_word_lists, test_tag_lists = build_corpus(test_path, make_vocab=False)
bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)


test_word_lists = [['上', '海', '浦', '东']]
test_tag_lists = []
length = len(test_word_lists[0])
temp = ['O' for i in range(length)]
test_tag_lists.append(temp)


def rnn_pred2(test_word_lists, test_tag_lists):
    # start = time.time()
    # vocab_size = len(bilstm_word2id)
    # out_size = len(bilstm_tag2id)
    path = "ckpts/rnn.pkl"
    model = joblib.load(path)
    # model = RNN_Model(vocab_size, out_size)
    # model.train(train_word_lists, train_tag_lists,
    #             dev_word_lists, dev_tag_lists, bilstm_word2id, bilstm_tag2id)
    # save_model(model, "./ckpts/rnn.pkl")
    # print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    # test_word_lists = test_word_lists[0:1]
    # test_tag_lists = test_tag_lists[0:1]
    print(test_word_lists)
    print(test_tag_lists)
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


res = rnn_pred2(test_word_lists, test_tag_lists)
print(res)
