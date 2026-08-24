import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import TrainDataLoader
from model import Net
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load only directed edges
def load_edges(path_d):
    index_d_src, index_d_dst = [], []
    d_edge_n = 0
    for line in open(path_d):
        s, t = map(int, line.strip().split())
        index_d_src.append(s)
        index_d_dst.append(t)
        d_edge_n += 1
    return (
        torch.tensor([index_d_src, index_d_dst], dtype=torch.long).to(device),
        d_edge_n
    )

def train(args):
    edge_index_directed, d_edge_n = load_edges("data/ednet/direct_graph.txt")
    args.d_edge_n = d_edge_n

    os.makedirs('model', exist_ok=True)
    net = Net(args).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = TrainDataLoader()

    # Fixed epoch training
    for epoch in range(args.epoch_n):
        net.train()
        loader.reset()
        total_loss = 0
        step = 0

        while not loader.is_end():
            step += 1
            stu, exer, qn, lab, qd = loader.next_batch()
            stu, exer, qn, lab, qd = [x.to(device) for x in (stu, exer, qn, lab, qd)]
            optimizer.zero_grad()

            logits, KL, _ = net(stu, exer, qn, qd, edge_index_directed)
            loss_pred = criterion(logits.view(-1), lab.float())
            loss = loss_pred + args.lambda_kl * KL
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epoch_n} | Loss={total_loss/step:.4f}")

    # Save model after training completes
    torch.save(net.state_dict(), 'model/best_model.pt')
    print(f"\nTraining complete! Model saved to model/best_model.pt")
    print(f"Total training epochs: {args.epoch_n}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epoch_n', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--lambda_kl', type=float, default=1e-3)
    p.add_argument('--student_n', type=int, default=672)
    p.add_argument('--exer_n', type=int, default=26)
    p.add_argument('--knowledge_n', type=int, default=16)
    p.add_argument('--d_edge_n', type=int, default=0)
    p.add_argument('--node_features', type=int, default=8)
    p.add_argument('--edge_features', type=int, default=8)
    p.add_argument('--num_hidden', type=int, default=16)
    p.add_argument('--num_convs', type=int, default=1)
    args = p.parse_args()
    train(args)