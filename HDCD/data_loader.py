import json
import torch

class TrainDataLoader(object):
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        data_file = 'data/ednet/train_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        d_edges = []
        with open("data/ednet/direct_graph.txt", 'r') as file:
            for line in file:
                source, target = map(int, line.split())
                d_edges.append((source, target))
        self.knowledge_dim = int(knowledge_n)
        self.d_edges = d_edges
        self.d_edge_dim = int(len(d_edges))

    def next_batch(self):
        if self.is_end():
            return None, None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, input_d_edge_embs, ys = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            d_edge_emb = [0.] * self.d_edge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
                for index, (source, target) in enumerate(self.d_edges):
                    if knowledge_code == source or knowledge_code == target:
                        d_edge_emb[index] = 1
            y = log['score']
            input_stu_ids.append(log['user_id'])
            input_exer_ids.append(log['exer_id'])
            input_knowledge_embs.append(knowledge_emb)
            input_d_edge_embs.append(d_edge_emb)
            ys.append(y)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowledge_embs), torch.LongTensor(ys), torch.LongTensor(input_d_edge_embs)

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0


class TestDataLoader(object):
    def __init__(self):
        self.batch_size = 16
        self.ptr = 0
        self.data = []
        data_file = 'data/ednet/test_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        d_edges = []
        with open("data/ednet/direct_graph.txt", 'r') as file:
            for line in file:
                source, target = map(int, line.split())
                d_edges.append((source, target))
        self.knowledge_dim = int(knowledge_n)
        self.d_edges = d_edges
        self.d_edge_dim = int(len(d_edges))

    def next_batch(self):
        if self.is_end():
            return None, None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, input_d_edge_embs, ys = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            d_edge_emb = [0.] * self.d_edge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
                for index, (source, target) in enumerate(self.d_edges):
                    if knowledge_code == source or knowledge_code == target:
                        d_edge_emb[index] = 1
            y = log['score']
            input_stu_ids.append(log['user_id'])
            input_exer_ids.append(log['exer_id'])
            input_knowledge_embs.append(knowledge_emb)
            input_d_edge_embs.append(d_edge_emb)
            ys.append(y)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowledge_embs), torch.LongTensor(ys), torch.LongTensor(input_d_edge_embs)

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0