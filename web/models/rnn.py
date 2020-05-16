'''
@Autor: xujiahuan
@Date: 2020-05-03 22:52:44
@LastEditors: xujiahuan
@LastEditTime: 2020-05-04 16:28:34
'''
import torch
from .config import RNNConfig, TrainingConfig
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .util import tensorized, sort_by_lengths, cal_loss
import torch.optim as optim
from copy import deepcopy


class RNN_Model(object):
    def __init__(self, vocab_size, out_size):
        """功能：对RNN的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类"""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = RNNConfig.emb_size
        self.hidden_size = RNNConfig.hidden_size
        self.model = RNN(vocab_size, self.emb_size,
                         self.hidden_size, out_size).to(self.device)
        self.cal_loss_func = cal_loss
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind+B]
                batch_tags = tag_lists[ind:ind+B]

                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)
        for i in range(len(lengths)):
            if lengths[i] == 0:
                lengths[i] = 1
        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)
                for i in range(len(lengths)):
                    if lengths[i] == 0:
                        lengths[i] = 1
                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        # print(len(word_lists))
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        self.best_model.eval()
        # print(lengths)
        # print(tag_lists)
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            for j in range(lengths[i]):
                tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, bidirectional=True)
        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        # print("emb", emb)
        # print("lengths:", lengths)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.rnn(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]
        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        # lengths = lengths[:-14]
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        print(logits)
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
