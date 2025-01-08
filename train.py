import os
import numpy
import pandas as pd
import argparse
import time
import torch
import torch.optim as optim
import yaml
from Auc import label_classification
from yaml import SafeLoader
from model import DFGCNN
from augment import aug
from loss import Loss
from utils import load, seed_everything


def train(model1, model2, loss, feat, graph, edge_drop_prob1, edge_drop_prob2, feature_mask_prob1, feature_mask_prob2):
    model1.train()
    model2.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    graph1, feat1 = aug(graph, feat, feature_mask_prob1, edge_drop_prob1)
    graph2, feat2 = aug(graph, feat, feature_mask_prob2, edge_drop_prob2)
    z1 = model1(feat1, graph1)
    z2 = model2(feat2, graph2)
    los1 = loss(z1, z2, mean=True)
    los1.backward()
    optimizer1.step()
    optimizer2.step()

    return los1.item()


def test(model, x, edge_index, y, train_mask, test_mask):
    model.eval()
    edge_index = edge_index.add_self_loop()
    z = model(x, edge_index)
    return label_classification(z, y, train_mask, test_mask, ratio=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFGCNN')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')

    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    torch.manual_seed(config['seed'])
    seed_everything(seed=config['seed'])
    # learning_rate = config['learning_rate']
    d1 = config['num_hidden']
    learning_rate=0.001
    d2 = config['num_proj_hidden']
    p_er1 = config['drop_edge_rate_1']
    p_er2 = config['drop_edge_rate_2']
    p_fm1 = config['drop_feature_rate_1']
    p_fm2 = config['drop_feature_rate_2']
    t = config['tau']
    epochs = config['num_epochs']
    # weight_decay = config['weight_decay']
    fussy = config['fussy']
    weight_decay=0.05


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lst = ['learning_rate', 'weight_decay', 'Micro-F1']
    save_csv = './result/' + "_DFGCNN.csv"
    pd.DataFrame(columns=lst).to_csv(save_csv, index=False)
    for learning_rate in 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005:
        for weight_decay in 0, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01:
            try:
                if args.dataset == 'Cora':
                    dataname = 'cora'
                    graph, feat, labels, train_mask, test_mask = load(dataname)
                num_in = feat.shape[1]
                model1 = DFGCNN(num_in, d1=d1, d2=d2, fussy=fussy)
                model2 = DFGCNN(num_in, d1=d1, d2=d2, fussy=fussy)
                model1.to(device)
                model2.to(device)

                optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)
                optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)

                loss1 = Loss(t)

                start = time.time()
                prev = start
                for epoch in range(1, epochs + 1):
                    loss = train(model1, model2, loss1, feat, graph, p_er1, p_er2, p_fm1, p_fm2)
                    if numpy.isnan(loss):
                        raise ValueError("Loss is NaN")
                    now = time.time()
                    print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                          f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                    prev = now
                print("=== Final ===")
                result1 = test(model1, feat, graph, labels, train_mask, test_mask)
                out = [learning_rate, weight_decay, result1]
                pd.DataFrame([out]).to_csv(save_csv, index=False, mode='a+', header=False)
            except ValueError as e:
                name = f'{learning_rate}-{weight_decay}'
                save_csv = './result/' + "-" + name + "_DFGCNN.csv"
                pd.DataFrame([e]).to_csv(save_csv, index=False, mode='a+', header=False)
                continue
