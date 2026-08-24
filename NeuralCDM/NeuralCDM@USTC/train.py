import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
from sklearn.metrics import roc_auc_score
import time
import warnings

from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# =======================================

def train(source_path, dataset, student_n, exer_n, knowledge_n, device, epoch_n):
    data_loader = TrainDataLoader(source_path, dataset)
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        start = time.time()
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        end = time.time()
        print('TIME:%.5f' %(end - start))

        save_dir = source_path + '/NeuralCDM/NeuralCDM@USTC/model_epoch/' + dataset
        os.makedirs(save_dir, exist_ok=True)
        save_snapshot(net, save_dir + '/model_epoch' + str(epoch + 1))


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    device = torch.device('cpu')
    epoch_n = 20
    print(f"Using device: {device}, Epochs: {epoch_n}")
    # ========================================

    DATASET = input("\nplease choose a dataset: [FrcSub, Junyi-s, ednet, assist09, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Junyi-s', 'ednet', 'assist09', 'Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()

    with open(BASE_DIR + '/NeuralCDM/NeuralCDM@USTC/configs/' + DATASET + '/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    train(BASE_DIR, DATASET, student_n, exer_n, knowledge_n, device, epoch_n)