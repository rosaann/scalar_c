from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import torch


def get_last_checkpoint(checkpoint_dir):
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  if checkpoints:
    return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
  return None


def get_initial_checkpoint(check_dir):
  checkpoint_dir = os.path.join(check_dir, 'checkpoint')
  return get_last_checkpoint(checkpoint_dir)


def get_checkpoint(check_dir, name):
  checkpoint_dir = os.path.join(check_dir, 'checkpoint')
  return os.path.join(checkpoint_dir, name)

def get_model_saved(dir, n_id):
    checkpoint_dir = os.path.join(dir,  'checkpoint')
    for checkpoint in os.listdir(checkpoint_dir):
        num = checkpoint.replace('epoch_', '')
        num = num.replace('.pth', '')
        if int(num) == n_id:
            return os.path.join(checkpoint_dir, checkpoint)

def copy_last_n_checkpoints(check_dir, n, name):
  checkpoint_dir = os.path.join(check_dir, 'checkpoint')
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  checkpoints = sorted(checkpoints)
  for i, checkpoint in enumerate(checkpoints[-n:]):
    shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                    os.path.join(checkpoint_dir, name.format(i)))


def load_checkpoint(model, optimizer, checkpoint):
  print('load checkpoint from', checkpoint)
  checkpoint = torch.load(checkpoint)
 # print('check ', checkpoint)
 # return

  checkpoint_dict = {}
  for k, v in checkpoint['state_dict'].items():
    if 'num_batches_tracked' in k:
      continue
    if k.startswith('module.'):
      if True:
        checkpoint_dict[k[7:]] = v
      else:
        checkpoint_dict['feature_extractor.' + k[7:]] = v
    else:
      if True:
        checkpoint_dict[k] = v
      else:
        checkpoint_dict['feature_extractor.' + k] = v

  model.load_state_dict(checkpoint_dict) #, strict=False)

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

  step = checkpoint['step'] if 'step' in checkpoint else -1
  last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

  return last_epoch, step


def save_checkpoint(check_dir, model, optimizer, epoch, step, weights_dict=None, name=None):
  checkpoint_dir = os.path.join(check_dir, 'checkpoint')

  if name:
    checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
  else:
    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(epoch))

  if weights_dict is None:
    weights_dict = {
      'state_dict': model.state_dict(),
      'optimizer_dict' : optimizer.state_dict(),
      'epoch' : epoch,
      'step' : step,
    }
  torch.save(weights_dict, checkpoint_path)
