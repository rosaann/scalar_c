#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:34:09 2019

@author: zl
"""
from models.linknet import LinkNet
import torch
import utils
import argparse
from models.model_factory import get_model
from losses.loss_factory import get_loss
from optimizers.optimizer_factory import get_optimizer
from dataset.dataset_factory import get_test_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='airbus')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()
def gen_test_content_list():
   #id,molecule_name,atom_index_0,atom_index_1
    df_test = pd.read_csv(os.path.join('data', 'test.csv'))
    num = df_test.shape[0]

    test_content_list = {}
    for i in tqdm.tqdm(range(num)):
        c_id = df_test.get_value(i, 'id')
        molecule_name = df_test.get_value(i, 'molecule_name')
        atom_index_0 = df_test.get_value(i, 'atom_index_0')
        atom_index_1 = df_test.get_value(i, 'atom_index_1')
        data = {'id':c_id, 'atom_index_0':atom_index_0, 'atom_index_1': atom_index_1}
        if molecule_name not in test_content_list.keys():
            test_content_list[molecule_name] = []
        test_content_list[molecule_name].append(data)
        
     return test_content_list       
            
    
def test_segmenter_single_epoch(config, model, dataloader):
   # model.eval()
    test_content_list = gen_test_content_list()
    torch.set_printoptions(threshold=1000000)
   
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    
    total_loss = 0
    result = []
    for i, data in tbar:
        images = data['data']
        mnames = data['name']
        
        if torch.cuda.is_available():
            images = images.cuda().float()
            
        
        binary_masks = model(images)
        mask, mname in zip(binary_masks, mnames)
            test_content = test_content_list[mname]
            
            for t in test_content:
                t_id = t['id']
                atom_index_0 = t['atom_index_0']
                atom_index_1 = t['atom_index_1']
                scale = mask[int(atom_index_0), int(atom_index_1)]
                result.append({'id':t_id, 'scalar_coupling_constant' : scale})
                
    test_pd = pd.DataFrame.from_records(result, columns=['id', 'scalar_coupling_constant'])
    output_filename = os.path.join('data', 'result.csv')
    test_pd.to_csv(output_filename, index=False)
def test_segmenter(config, model, test_dataloader):
    
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
    
    test_segmenter_single_epoch(config, model, test_dataloader)
def main():
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    
    model_segmenter = LinkNet(1)
    if torch.cuda.is_available():
        model_segmenter = model_segmenter.cuda()
    
    criterion_segmenter = get_loss(config.loss_segmenter)
    optimizer_segmenter = get_optimizer(config.optimizer_segmenter.name, model_segmenter.parameters(), config.optimizer_segmenter.params)
    ####
    checkpoint = utils.checkpoint.get_model_saved(config, 299)
    best_epoch, step = utils.checkpoint.load_checkpoint(model_segmenter, optimizer_segmenter, checkpoint)
    

    test_segmenter_dataloaders = get_test_dataloader(600)
    
    test_segmenter(config, model_segmenter, test_segmenter_dataloaders)

if __name__ == '__main__':
    main()