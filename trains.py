# -*- encoding: utf-8 -*-
import os
import time
import yaml
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging
import random
import utils
import torch.nn as nn
from data.data_loader import data_loader
from models.model import SelfGNN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

config_filename = f'configs/cfg.yaml'
configs = yaml.load(open(config_filename), Loader=yaml.FullLoader)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=configs['dataset'][configs['dataset_idx']], 
                type=str, help='the configuration to use')
args = parser.parse_args()

set_seed(configs["seed"])

logpath = os.path.dirname(__file__) + f'/logs/{time.time()}'
os.makedirs(logpath)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logpath + '/loss.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

tbpath = os.path.dirname(__file__) + f'/tblog/{time.time()}'
os.makedirs(tbpath)
writer = SummaryWriter(tbpath)

if __name__ == '__main__':
    dataset = args.dataset
    print(dataset)
    train_loader, test_loader, scaler = data_loader(configs["batch_size"], dataset=dataset,\
              rep_times=configs['rep_times'])

    device = torch.device(
        configs["device"]) if torch.cuda.is_available() else torch.device('cpu')

    net = SelfGNN(configs)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=configs["lr"], weight_decay=configs["wd"])
    net.train()
    save_path = os.path.dirname(__file__) + f'/model_save/{dataset}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    retrain = configs['retrain']
    ckpt = [f'best_model_{dataset}.pth']
    if not retrain and ckpt:
        ckpt_path = os.path.join(save_path, sorted(ckpt)[0])
        ckpt = torch.load(ckpt_path)
        net.load_state_dict(ckpt['net_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        current_epoch = ckpt['epoch'] + 1
    else:
        net.apply(utils.weight_init)
        current_epoch = 0
    min_loss = 1e10
    ttl_epochs = configs['epochs']
    mseloss = nn.L1Loss().to(device)
    torch.save(
    {
        'epoch': 0,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },
    f"model_save/model_init_{0}th.pth")
    for epoch in range(current_epoch, ttl_epochs):
        ttl_loss = 0
        krig_ttl = 0
        utils.adjust_learning_rate(optimizer, epoch, configs["lr"], \
            configs["decay_steps"], configs['decay_rates'])
        for j, data in enumerate(train_loader):
            data = data.to(device)
            loss, acc = net(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ttl_loss += loss.data
        if (ttl_loss.item() / (j + 1)) < min_loss:
            min_loss = (ttl_loss.item() / (j + 1))
            torch.save(
                {
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                f"model_save/{dataset}/model_{min_loss:.4f}_{epoch}th.pth"
            )
            torch.save(
                {
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                f"model_save/{dataset}/best_model_{dataset}.pth"
            )

        writer.add_scalar("loss/Train Loss", (ttl_loss / (j + 1)).data, epoch)
        logger.info(f'epoch: {epoch}, loss={(ttl_loss / (j+1)).item():.4f}')