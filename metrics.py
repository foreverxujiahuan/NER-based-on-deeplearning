'''
@Autor: xujiahuan
@Date: 2020-04-21 15:27:58
@LastEditors: xujiahuan
@LastEditTime: 2020-04-21 20:33:56
'''


class Metrics:
    def __init__(self, pred_tag_lists, golden_tag_lists):
        self.pred_tag_lists = pred_tag_lists
        self.golden_tag_lists = golden_tag_lists
        pred = self.change_data(pred_tag_lists)
        golden = self.change_data(golden_tag_lists)
        TP = 0
        for t in pred:
            if t in golden:
                TP = TP + 1
        self.precision = TP/len(pred)
        self.recall = TP/len(golden)
        print("P %.4f" % self.precision)
        print("P %4f" % self.recall)
        self.f1 = 2*(self.precision*self.recall)/(self.precision+self.recall)

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def aget_f1(self):
        return self.f1

    def change_data(self, lists):
        length = len(lists)
        res = []
        for i in range(length):
            t = lists[i]
            flag = 0
            start = 0
            end = 0
            for j in range(len(t)):
                if t[j] != 'O':
                    if t[j] == 'B-ORG':
                        flag = 1
                        start = j
                    if t[j] == 'B-PER':
                        flag = 2
                        start = j
                    if t[j] == 'B-LOC':
                        flag = 3
                        start = j
                else:
                    end = j
                    if flag != 0:
                        res.append([i, start, end, flag])
                    flag = 0
        return res
