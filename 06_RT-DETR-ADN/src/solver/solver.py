"""by lyuwenyu
"""

import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
from typing import Dict

from src.misc import dist
from src.core import BaseConfig


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        
        self.cfg = cfg 

    def setup(self, ):
        '''Avoid instantiating unnecessary classes 
        '''
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        # 2024.05.15 @hslee 
        # model definition
        self.super_config = [False, False, False, False]
        self.base_config = [True, True, True, True]
        
        model = cfg.model.to(device) # cfg.model = (YAMLConfig Class).(method) in yaml_config.py
        self.model = dist.warp_model(model, cfg.find_unused_parameters, cfg.sync_bn)
        print(f"self.model (in solver.py): \n{self.model}")
        
        '''
        (rt detr with original pretrained ResNet50) number of params: 42862860
        (rt detr with my pretrained ResNet50ADN)    number of params: 42862860
            -> setting 1 : (in resnetADN.py) norm_layer : FrozenBatchNorm2d
            -> setting 2 : (in solver.py)    backbone.conv1.weight.requires_grad = False (9408)
        '''
        
        # 2024.06.04 @hslee 
        # setting 2 : if pretrained ResNet50ADN, set backbone.conv1.weight to requires_grad=False
        # because original pretrained model has backbone.conv1.weight requires_grad=False
        if cfg.yaml_cfg['RTDETR']['backbone'] == 'PResNet' :
            pass
        elif cfg.yaml_cfg['RTDETR']['backbone'] == 'SwinTransformer' :
            pass
            # # set parameter named in 'pre' to requires_grad=False
            # for name, param in self.model.named_parameters():
            #     if 'pre' in name:
            #         param.requires_grad = False
        else :
            # multi GPU
            if dist.is_parallel(self.model):
                self.model.module.backbone.conv1.weight.requires_grad = False
            # single GPU
            else :
                self.model.backbone.conv1.weight.requires_grad = False
        
        # print the all parameters in the model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
            else :
                print(f"{name} (not requires_grad)")
        
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def train(self, ):
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)


    def eval(self, ):
        self.setup()
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)


    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state


    def load_state_dict(self, state):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')


    def save(self, path):
        '''save state
        '''
        state = self.state_dict()
        dist.save_on_master(state, path)


    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        raise NotImplementedError('')

    def val(self, ):
        raise NotImplementedError('')
