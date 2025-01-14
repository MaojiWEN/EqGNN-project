import argparse
import torch
from torch_geometric.loader import DataLoader
from eth_ucy.eth_new import ETHNew
from baseline import ESTAG
import os
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--past_length', type=int, default=8, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--future_length', type=int, default=12, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--epoch_decay', type=int, default=2, metavar='N',
                    help='number of epochs for the lr decay')
parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='N',
                    help='the lr decay ratio')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--channels', type=int, default=64, metavar='N',
                    help='number of channels')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                    help='0 nor sweep, 1 sweep, 2 sweep small')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--subset', type=str, default='eth',
                    help='Name of the subset.')
parser.add_argument('--model_save_dir', type=str, default='eth_ucy/ESTAG',
                    help='Name of the subset.')
parser.add_argument('--scale', type=float, default=1, metavar='N',
                    help='dataset scale')
parser.add_argument("--apply_decay", action='store_true')
parser.add_argument("--res_pred", action='store_true')
parser.add_argument("--supervise_all", action='store_true')
parser.add_argument('--model_name', type=str, default='eth_ckpt_best', metavar='N',
                    help='dataset scale')
parser.add_argument('--test_scale', type=float, default=1, metavar='N',
                    help='dataset scale')
parser.add_argument("--test", action='store_true')
parser.add_argument("--vis", action='store_true')

time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
args.cuda = True

device = torch.device("cuda" if args.cuda else "cpu")
# loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

num_nodes = 0
if args.subset == 'zara1_main.sh':
    args.channels = 128
else:
    args.channels = 64


if args.subset == 'eth':
    args.test_scale = 1.6

if args.subset == 'univ':
    num_nodes = 2
elif args.subset == 'zara1_main.sh':
    num_nodes = 3
else:
    num_nodes = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new


def main():
    # seed = 861
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('The seed is :', seed)


    dataset_train = ETHNew(args.subset, args.past_length, args.future_length, traj_scale=1., phase='training',
                           return_index=True)
    dataset_test = ETHNew(args.subset, args.past_length, args.future_length, traj_scale=1., phase='testing',
                          return_index=True)

    # 使用 torch_geometric 的 DataLoader
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of testing samples: {len(dataset_test)}")

    model = ESTAG(num_past=args.past_length, num_future=args.future_length, in_node_nf=1, in_edge_nf=1, hidden_nf=args.channels
                  , fft=True, eat=True, device=device, n_layers=2, n_nodes=num_nodes, with_mask=True, nodes_att_dim=args.past_length-1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.test:
        model_path = args.model_save_dir + '/' + args.model_name + '.pth.tar'
        print('Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=False)
        test_loss, ade, fde = test(model, loader_test, 0, model.device)
        print('ade:', ade, 'fde:', test_loss)


    results = {'epochs': [], 'losess': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    lr_now = args.lr
    for epoch in range(0, args.epochs):
        if args.apply_decay:
            if epoch % args.epoch_decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
        train(model, optimizer, epoch, loader_train)
        if epoch % args.test_interval == 0:
            test_loss, ade, fde = test(model, loader_test, 0, model.device)
            results['epochs'].append(epoch)
            results['losess'].append(test_loss)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_ade = ade
                best_epoch = epoch

                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                subset_dir = os.path.join(args.model_save_dir, str(args.subset))
                if not os.path.exists(subset_dir):
                    os.makedirs(subset_dir)
                file_path = os.path.join(subset_dir, 'ckpt_best.pth.tar')
                # 保存模型
                torch.save(state, file_path)

            print("Best Test Loss: %.5f \t Best ade: %.5f \t Best fde: %.5f \t Best epoch %d" % (best_test_loss, best_ade, fde, best_epoch))


            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            subset_dir = os.path.join(args.model_save_dir, str(args.subset))
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir)
            file_path = os.path.join(subset_dir, 'ckpt_'+str(epoch)+'.pth.tar')
            torch.save(state, file_path)

    return best_val_loss, best_test_loss, best_epoch


constant = 1


def get_valid_mask2(num_valid, agent_num):
    batch_size = num_valid.shape[0]
    valid_mask = torch.zeros((batch_size, agent_num))
    for i in range(batch_size):
        valid_mask[i, :num_valid[i]] = 1
    return valid_mask.unsqueeze(-1).unsqueeze(-1)


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    # 初始化结果字典，确保包含 'ade' 和 'fde'
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'ade': 0, 'fde': 0}

    for batch_idx, data in enumerate(loader):
        if data is not None:
            # 获取数据
            h = data.h.to(model.device)  # [batch_size * num_nodes, num_past, 2]
            x_past = data.x_past.to(model.device)  # [batch_size * num_nodes, past_length, 2]
            x_future = data.x_future.to(model.device)  # [batch_size * num_nodes, future_length, 2]
            edge_index = data.edge_index.to(model.device)  # [2, num_edges]
            edge_attr = data.edge_attr.to(model.device)  # [num_edges, 1]

            # 模型预测
            loc_pred = model(h=h, x=x_past, edges=edge_index, edge_attr=edge_attr)
            loc_pred = loc_pred.permute(1, 0, 2)

            loc_future_gt = x_future
            ade = torch.mean(torch.norm(loc_pred - loc_future_gt, dim=-1))  # 标量
            fde = torch.mean(torch.norm(loc_pred[:, -1, :] - loc_future_gt[:, -1, :], dim=-1))  # 标量

            loss = ade  # 在这个实现中我们只使用 ADE 作为优化目标

            if backprop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 记录损失
            res['loss'] += loss.item() * x_past.size(0)
            res['ade'] += ade.item() * x_past.size(0)
            res['fde'] += fde.item() * x_past.size(0)
            res['counter'] += x_past.size(0)

    # 计算平均损失
    avg_loss = res['loss'] / res['counter']
    avg_ade = res['ade'] / res['counter']
    avg_fde = res['fde'] / res['counter']
    print(f'Epoch {epoch}, Avg Loss: {avg_loss:.5f}, Avg ADE: {avg_ade:.5f}, Avg FDE: {avg_fde:.5f}')

    return avg_loss, avg_ade, avg_fde


def test(model, loader, epoch, device):
    model.eval()

    # 初始化结果字典，确保包含 'ade' 和 'fde'
    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'ade': 0, 'fde': 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if data is not None:
                # 获取数据
                h = data.h.to(device)  # [batch_size * num_nodes, num_past, 2]
                x_past = data.x_past.to(device)  # [batch_size * num_nodes, past_length, 2]
                x_future = data.x_future.to(device)  # [batch_size * num_nodes, future_length, 2]
                edge_index = data.edge_index.to(device)  # [2, num_edges]
                edge_attr = data.edge_attr.to(device)  # [num_edges, 1]

                # 模型预测
                loc_pred = model(h=h, x=x_past, edges=edge_index, edge_attr=edge_attr)
                loc_pred = loc_pred.permute(1, 0, 2)

                loc_future_gt = x_future
                ade = torch.mean(torch.norm(loc_pred - loc_future_gt, dim=-1))  # 标量
                fde = torch.mean(torch.norm(loc_pred[:, -1, :] - loc_future_gt[:, -1, :], dim=-1))  # 标量

                # 记录损失
                res['loss'] += ade.item() * x_past.size(0)
                res['ade'] += ade.item() * x_past.size(0)
                res['fde'] += fde.item() * x_past.size(0)
                res['counter'] += x_past.size(0)

    # 平均损失和误差
    avg_loss = res['loss'] / res['counter']
    avg_ade = res['ade'] / res['counter']
    avg_fde = res['fde'] / res['counter']

    print(f'Test Epoch {epoch}, Avg Loss: {avg_loss:.5f}, Avg ADE: {avg_ade:.5f}, Avg FDE: {avg_fde:.5f}')

    return avg_loss, avg_ade, avg_fde





if __name__ == "__main__":
    main()