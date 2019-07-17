#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:33:54 2019

@author: zl
"""

import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
#from dataset.dataset_factory import get_gnu_dataloader
from models.gnc import Regression_X1
from dataset.dgl_dataset_factory import get_gnu_dataloader
from transforms.transform_factory import get_transform
from models.model_factory import get_model
from losses.loss_factory import get_loss
from optimizers.optimizer_factory import get_optimizer
from schedulers.scheduler_factory import get_scheduler
from utils.utils import prepare_train_directories
import utils.config
from utils.checkpoint import *
from utils.metrics import *
from models.model_factory import get_model
from multiprocessing.pool import ThreadPool
from scipy import ndimage
import torch.nn as nn
#from tools.gen_gt_images import genBiImage
import torchvision.utils as vutils
import cv2
from torchvision import transforms
from utils.confusion_matrix import ConfusionMatrix
from utils.r2score import R2Score


def evaluate_segmenter_single_epoch(config, model, dataloader, criterion,
                          epoch, writer, postfix_dict, metrics):
    model.eval()

    with torch.no_grad():
        batch_size = config.train_segmenter.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

       # probability_list = []
       # label_list = []
        loss_list = []
      #  tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
      #  out_images_dir = './data/val_result/'
      #  print('total_step val ', total_step)
        rc_total_list = []
        rc_part_list = []
       
        mae_total_list = []
        mae_part_list = []
        for i, (images, gt) in enumerate(dataloader):
           # print('-------------uu------------')
            if torch.cuda.is_available():
      #      images = images.cuda().float()
                gt = gt.cuda().float()
            print('img ', images)
            print('gt ', gt)
            binary_masks = model(images)
            
            print('binary_masks ', binary_masks)
            loss = criterion(binary_masks, gt)
           # if i < 10:
            pred = binary_masks.data.cpu().numpy()
            gt = gt.cpu().numpy()
           # print('metrics ', metrics)
            rc_t, rc_part, msq_t, msq_part, mae_t, mae_part = metrics.count(gt, pred)
            rc_total_list.append(rc_t)
            rc_part_list.append(rc_part)
           # msq_total_list.append(msq_t)
          #  msq_part_list.append(msq_part)
            mae_total_list.append(mae_t)
            mae_part_list.append(mae_part)
            # measure accuracy and record loss
            loss_list.append(loss.item())
            

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
          #  tbar.set_description(desc)
          #  tbar.set_postfix(**postfix_dict)

        
      #  metrics.reset()
        
        log_dict = {}
       
        log_dict['loss'] = sum(loss_list) 
        log_dict['total_r2'] = sum(rc_total_list)/len(rc_total_list)
        log_dict['used_r2'] = sum(rc_part_list) / len(rc_part_list)
       # log_dict['msq_total'] = sum(msq_total_list) / len(msq_total_list)
       # log_dict['msq_part'] = sum(msq_part_list) / len(msq_part_list)
        log_dict['mae_total'] = sum(mae_total_list) / len(mae_total_list)
        log_dict['mae_part'] = sum(mae_part_list) / len(mae_part_list)

        for key, value in log_dict.items():
            if writer is not None:
             #   print('key ', key)
            #    print('value ', value)
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return metrics
def train_segmenter_single_epoch(config, model, dataloader, criterion, optimizer,
                       epoch, writer, postfix_dict):
    model.train()
    torch.set_printoptions(threshold=1000000)
    batch_size = config.train_segmenter.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
   # tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    
    total_loss = 0
    for i, (images, gt) in enumerate(dataloader):
     #  images = data['data']
     #   gt = data['gt']
       # paths = data['path']
        
        if torch.cuda.is_available():
      #      images = images.cuda().float()
            gt = gt.cuda().float()
        
        binary_masks = model(images)
        
     #   print('binary_masks ', binary_masks.shape, ' ',  binary_masks)
     #   print('gt ', gt.shape, ' ', gt )
        loss = criterion(binary_masks, gt)

            # measure accuracy and record loss
        total_loss += loss.item()

            # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
     #   print('binary_masks ', binary_masks.shape, ' mask ',binary_masks )
    #    remaining_ids = list(map(lambda path: path.split('/')[-1], paths))
    #    print('remaining_ids ', remaining_ids)
    #    results = postprocess_segmentation(pool, remaining_ids[:len(binary_masks)], binary_masks)
      #  print('logits ', logits.shape)
      #  print('labels ', labels.shape)

        if config.train_classifier.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train_classifier.num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()  

        f_epoch = epoch + i / total_step

        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
      #  tbar.set_description(desc)
      #  tbar.set_postfix(**postfix_dict)

    log_dict['lr'] = optimizer.param_groups[0]['lr']
    
    log_dict['loss'] = total_loss
    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('train/{}'.format(key), value, epoch)
            
def train_segmenter(config, model, train_dataloader, eval_dataloaders, criterion, optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train_segmenter.num_epochs
    
    metrics = R2Score()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    postfix_dict = {'train/lr': 0.0,
                    'train/acc': 0.0,
                    'train/loss': 0.0,
                    'val/f1': 0.0,
                    'val/acc': 0.0,
                    'val/loss': 0.0}

    f1_list = []
    best_f1 = 0.0
    best_f1_mavg = 0.0
    print('train_dataloader ', len(train_dataloader))
    print('eval_dataloaders ', len(eval_dataloaders))
    for epoch in range(start_epoch, num_epochs):
        # train phase
       # train_dataloader.setSplit('train')
        train_segmenter_single_epoch(config, model, train_dataloader,
                           criterion, optimizer, epoch, writer, postfix_dict)

        # val phase
      #  train_dataloader.setSplit('val')
        metrics = evaluate_segmenter_single_epoch(config, model, eval_dataloaders,
                                   criterion, epoch, writer, postfix_dict, metrics)

      #  if scheduler.name == 'reduce_lr_on_plateau':
      #    scheduler.step(f1)
      #  elif scheduler.name != 'reduce_lr_on_plateau':
      #    scheduler.step()

        utils.checkpoint.save_checkpoint(config.train_segmenter.dir, model, optimizer, epoch, 0)
        
        
    return {'f1': best_f1, 'f1_mavg': best_f1_mavg}
def run(config):
   # train_dir = config.train.dir
    
   # model_segmenter = get_model(config.model_segmenter.name)
    model_segmenter = Regression_X1() #NetX2()#LinkNet(1)
    if torch.cuda.is_available():
        model_segmenter = model_segmenter.cuda()
    criterion_segmenter = get_loss(config.loss_segmenter)
    optimizer_segmenter = get_optimizer(config.optimizer_segmenter.name, model_segmenter.parameters(), config.optimizer_segmenter.params)
    
    ####
    checkpoint_segmenter = get_initial_checkpoint(config.train_segmenter.dir)
    if checkpoint_segmenter is not None:
        last_epoch, step = load_checkpoint(model_segmenter, optimizer_segmenter, checkpoint_segmenter)
    else:
        last_epoch, step = -1, -1

    print('from segmenter checkpoint: {} last epoch:{}'.format(checkpoint_segmenter, last_epoch))
  #  scheduler = get_scheduler(config, optimizer, last_epoch)
    print('config.train ', config.train)
    writer = SummaryWriter(config.train.writer_dir)
    
    scheduler = 'none'
  #  train_classifier_dataloaders = get_dataloader(config.data_classifier, './data/data_train.csv',config.train_classifier.batch_size, 'train',config.transform_classifier.num_preprocessor, get_transform(config.transform_classifier, 'train'))
  #  eval_classifier_dataloaders = get_dataloader(config.data_classifier, './data/data_val.csv',config.eval_classifier.batch_size, 'val', config.transform_classifier.num_preprocessor, get_transform(config.transform_classifier, 'val'))
  #  test_dataloaders = get_dataloader(config.data_classifier,'./data/data_test.csv', get_transform(config, 'test'))
       
  #  train_classifier(config, model_classifier, train_classifier_dataloaders,eval_classifier_dataloaders, criterion_classifier, optimizer_classifier, scheduler,
  #        writer, last_epoch+1)
    
    criterion_segmenter = nn.MSELoss()
    
    train_segmenter_dataloaders = get_gnu_dataloader(config.train_segmenter.batch_size, 'train' )
    
    eval_segmenter_dataloaders = get_gnu_dataloader(config.train_segmenter.batch_size, 'val')
  
    train_segmenter(config, model_segmenter, train_segmenter_dataloaders, eval_segmenter_dataloaders, criterion_segmenter, optimizer_segmenter, scheduler,
          writer, last_epoch+1)

def parse_args():
    parser = argparse.ArgumentParser(description='airbus')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()
    
def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('train scale_c Classification Challenge.')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
