import json
import os

import torch


class Recoder:

    def __init__(self, file_name, args, metric_num, positive_order=True):
        self.args = args
        self.file_name = file_name
        self.best_metric = [0 for _ in range(metric_num)]
        self.order = positive_order
        if not positive_order:
            self.best_metric = [100000 for _ in range(metric_num)]
        self.data = []

    def record(self, now_metric, state_dict=None, data=None):
        if self.compare(now_metric, self.best_metric):
            self.best_metric = now_metric
            if state_dict is not None:
                self.save(state_dict, 'model')
            if data is not None:
                self.data.append(data)
            return True
        return False

    def get_error_pred(self, ground_data, pred_data):
        data = []
        for i1, j1 in zip(ground_data, pred_data):
            c = 0
            for m, n in zip(i1['consistency'], j1):
                if m == n:
                    c += 1
            if c != 3:
                data.append({'knowledge_base': i1['knowledge_base'], 'history': i1['history'],
                             'last_response': i1['last_response'], 'consistency': str(j1)})
        return data

    def print_output(self):
        fo = open(self.file_name + "out.txt", "w")
        fo.write(self.print_method(self.data))
        fo.close()

    def print_method(self, s):
        return str(json.dumps(s))

    def compare(self, now, best):
        for i, j in zip(now, best):
            if self.order and i <= j:
                return False
            if not self.order and i >= j:
                return False
        return True

    def save(self, state_dict, name):
        file = "{}/{}.pkl".format(self.args.dir.output, name)
        if not os.path.exists(self.args.dir.output):
            os.makedirs(self.args.dir.output)
        state = {"models": state_dict}
        torch.save(state, file)


if __name__ == '__main__':
    r = Recoder("../../out/", None, 1)
    r.record([2])
