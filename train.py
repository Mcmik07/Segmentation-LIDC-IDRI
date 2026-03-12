import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from losses import BCEDiceLoss
from dataset import MyLidcDataset
from metrics import iou_score,dice_coef
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from Unet.WithAttention import AttentionUNet
from Unet.UnetTranswithAttention import UNetTransWithAttention
from UnetNested.Nested_Unet import NestedUNet
from Unet3Plus.Unet_3Plus import UNet3Plus
from DeepLabV3Plus.DeepLabV3Plus import DeepLabV3Plus
from ResnetPlus.ResnetPlus import ResUNetPlusPlus
from ResDSda_UNet.ResDSda_UNet import ResDSda_UNet

from TransUnet.TransUNet import TransUNet
from Segnet.Segnet import SegNet
from Unet.AttTransUNet import AttnTransUNet
from TransUnet_b.TransUnet_b import TransUnet_b
from DeepLabV3.deeplabv3 import DeepLabV3
from Unet3Plus.Atts_Res_dil_UNet3Plus import Atts_Res_dil_UNet3Plus
from Unet.As_AttRes_dil_UNet import As_AttRes_dil_UNet

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--name', default="AttentionUNet",
                        help='model name: UNET',choices=['UNET', 'AttentionUNet', 'UNet3Plus', 'UNetTransWithAttention',
                                                         'DeepLabV3Plus', 'ResUNetPlusPlus','ResDSda_UNet', 'TransUNet',
                                                         'SegNet', 'AttnTransUNet', 'TransUnet_b','DeepLabV3', 'Atts_Res_dil_UNet3Plus',
                                                         'As_AttRes_dil_UNet'])
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('--num_workers', default=2, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # data
    parser.add_argument('--augmentation',type=str2bool,default=True,choices=[True,False])



    config = parser.parse_args()

    return config


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    # Get configuration
    config = vars(parse_args())
    # Make Model output directory

    if config['augmentation']== True:
        file_name= config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] +'_base'
    os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    #save configuration
    with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    #criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model" )
    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=1)
    elif config['name'] == 'UNet3Plus':
        model = UNet3Plus(num_classes=1, input_channels=1)
    elif config['name'] == 'DeepLabV3Plus':
        model = DeepLabV3Plus(num_classes=1, input_channels=1)
    elif config['name'] == 'ResUNetPlusPlus':
        model = ResUNetPlusPlus(num_classes=1, input_channels=1)
    elif config['name'] == 'ResDSda_UNet':
        model = ResDSda_UNet(1, 1)
    elif config['name'] == 'AttentionUNet':
        model = AttentionUNet(img_ch=1, output_ch=1)
    elif config['name'] == 'UNetTransWithAttention':
        model = UNetTransWithAttention(n_channels=1, n_classes=1)
    elif config['name'] == 'TransUNet':
        model = TransUNet(img_dim=256,in_channels=1,out_channels=128,head_num=4,mlp_dim=512,block_num=8,patch_dim=16,class_num=1)
    elif config['name'] == 'SegNet':
        model = SegNet(input_nbr=1,label_nbr=1)
    elif config['name'] == 'AttnTransUNet':
        model = AttnTransUNet(input_channels=1, num_classes=1, base_filters=32, num_heads=8, ff_dim=512,
                              use_multiscale_fusion=False)
    elif config['name'] == 'TransUnet_b':
        model = TransUnet_b(num_classes=1, input_channels=1, feature_scale=1, num_heads=8, ff_dim=512)
    elif config['name'] == 'DeepLabV3':
        model = DeepLabV3(class_num=1)
    elif config['name'] == 'Atts_Res_dil_UNet3Plus':
        model = Atts_Res_dil_UNet3Plus(num_classes=1, input_channels=1, feature_scale=1, dilation=2)
    elif config['name'] == 'As_AttRes_dil_UNet':
        model = As_AttRes_dil_UNet(n_channels=1, n_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())


    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Directory of Image, Mask folder generated from the preprocessing stage ###
    # Write your own directory                                                 #
    IMAGE_DIR = "data/Image"                                       #
    MASK_DIR = "data/Mask"                                         #
    #Meta Information                                                          #
    meta = pd.read_csv("data/Meta/meta.csv")                    #
    ############################################################################
    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x: os.path.join(IMAGE_DIR, x + '.npy'))
    meta['mask_image'] = meta['mask_image'].apply(lambda x: os.path.join(MASK_DIR, x + '.npy'))

    train_meta = meta[meta['data_split']=='Train']
    val_meta = meta[meta['data_split']=='Validation']

    # Get all *npy images into list for Train
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])

    # Get all *npy images into list for Validation
    val_image_paths = list(val_meta['original_image'])
    val_mask_paths = list(val_meta['mask_image'])
    print("*"*50)
    print("The lenght of image: {}, mask folders: {} for train".format(len(train_image_paths),len(train_mask_paths)))
    print("The lenght of image: {}, mask folders: {} for validation".format(len(val_image_paths),len(val_mask_paths)))
    print("Ratio between Val/ Train is {:2f}".format(len(val_image_paths)/len(train_image_paths)))
    print("*"*50)



    # Create Dataset
    train_dataset = MyLidcDataset(train_image_paths, train_mask_paths,config['augmentation'])
    val_dataset = MyLidcDataset(val_image_paths,val_mask_paths,config['augmentation'])
    #test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    # Create Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=6)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)

    log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

    best_dice = 0
    trigger = 0

    # Crée un DataFrame vide avec des types bien définis
    log = pd.DataFrame({
        'epoch': pd.Series(dtype='int'),
        'lr': pd.Series(dtype='float'),
        'loss': pd.Series(dtype='float'),
        'iou': pd.Series(dtype='float'),
        'dice': pd.Series(dtype='float'),
        'val_loss': pd.Series(dtype='float'),
        'val_iou': pd.Series(dtype='float'),
        'val_dice': pd.Series(dtype='float')
    })

    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)


        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
            epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'],val_log['iou']))

        tmp = pd.Series([
            epoch,
            config['lr'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice']
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou','val_dice'])

        log = pd.concat([log, pd.DataFrame([tmp])], ignore_index=True)
        log.to_csv('model_outputs/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'model_outputs/{}/model.pth'.format(file_name))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

        scheduler.step(val_log['dice'])

        # 🔍 Afficher le learning rate après update
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

if __name__ == '__main__':
    main()
