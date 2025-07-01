import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
import dataloader
import DarkLighter_model as model
import Myloss
import numpy as np
from tqdm import tqdm

from utils.utils import get_lr 


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def fit_one_epoch(model, enhancer, yolo_loss, optimizer, train_loader, val_loader, epoch, Epoch, config):
    model.train()
    enhancer.train()

    loss_total = 0
    loss_dehaze_total = 0
    loss_det_total = 0

    criterion = nn.MSELoss()

    print("Start Training")
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{Epoch}", mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            images, targets, clearimgs = batch[0].cuda(), [ann.cuda() for ann in batch[1]], batch[2].cuda()

            optimizer.zero_grad()

            dark_and_clear = torch.cat([images, clearimgs], dim=0)

            det_out = model(dark_and_clear)            # Detection head
            enhanced, A, N = enhancer(images)           # Enhancement head

            loss_det = yolo_loss(det_out[0], targets)
            loss_dehaze = 50 * torch.mean(Myloss.L_color()(enhanced)) + \
                          10 * torch.mean(Myloss.L_cen(16, 0.6)(enhanced)) + \
                          0.001 * torch.norm(Myloss.perception_loss()(enhanced) - Myloss.perception_loss()(images)) + \
                          1600 * Myloss.L_ill()(A) + \
                          50 * torch.mean(Myloss.noise_loss()(N))

            total_loss = 0.2 * loss_det + 0.8 * loss_dehaze

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(enhancer.parameters()), config.grad_clip_norm)
            optimizer.step()

            loss_total += total_loss.item()
            loss_dehaze_total += loss_dehaze.item()
            loss_det_total += loss_det.item()

            pbar.set_postfix({
                'loss': loss_total / (iteration + 1),
                'dehaze_loss': loss_dehaze_total / (iteration + 1),
                'det_loss': loss_det_total / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    print("Finish Training")

    # Validation
    model.eval()
    print("Start Validation")
    val_loss_total = 0
    with tqdm(total=len(val_loader), desc=f"Val Epoch {epoch+1}/{Epoch}", mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            images, targets = batch[0].cuda(), [ann.cuda() for ann in batch[1]]

            with torch.no_grad():
                det_out = model(images)
                val_loss = yolo_loss(det_out[0], targets)

            val_loss_total += val_loss.item()
            pbar.set_postfix({'val_loss': val_loss_total / (iteration + 1)})
            pbar.update(1)

    print(f"Epoch {epoch+1}/{Epoch} || Total Loss: {loss_total:.3f} || Val Loss: {val_loss_total/len(val_loader):.3f}")

    if (epoch + 1) % config.snapshot_iter == 0 or (epoch + 1) == Epoch:
        torch.save({
            'model': model.state_dict(),
            'enhancer': enhancer.state_dict()
        }, os.path.join(config.snapshots_folder, f"Epoch{epoch+1}.pth"))


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_det = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False).cuda()
    enhancer = model.enhancer().cuda()

    model_det.apply(weights_init)
    enhancer.apply(weights_init)

    if config.load_pretrain:
        state_dict = torch.load(config.pretrain_dir)
        enhancer.load_state_dict(state_dict['enhancer'])

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    val_dataset = dataloader.lowlight_loader(config.val_images_path, mode='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    yolo_loss = nn.MSELoss()  # Replace with your YOLO loss if needed

    optimizer = torch.optim.Adam(list(model_det.parameters()) + list(enhancer.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        fit_one_epoch(model_det, enhancer, yolo_loss, optimizer, train_loader, val_loader, epoch, config.num_epochs, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lowlight_images_path', type=str, default="data/train/")
    parser.add_argument('--val_images_path', type=str, default="data/val/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch168.pth")

    config = parser.parse_args()

    os.makedirs(config.snapshots_folder, exist_ok=True)
    train(config)
