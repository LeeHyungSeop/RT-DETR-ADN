"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale # [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        
        
    # [True, True, True, True]
    def forward(self, x, targets=None, skip=None):
        # print(f"[in RTDETR.forward] skip : {skip}")
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            # sz : random size from multi_scale [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
            x = F.interpolate(x, size=[sz, sz])
        
        # 2024.05.23 @hslee    
        # print(f"\traw image size : {x.shape}")
        '''
            x : input
                torch.Size([bs, 3, sz, sz])
        '''
        
        x = self.backbone(x, skip=skip)
        # print(f"\t(after RTDETR.backbone)")
        # for i in range(len(x)):
        #     print(f"\tx[{i}] : {x[i].shape}")
        '''
                
            x[0] : torch.Size([4,  512, 4h, 4w])
            x[1] : torch.Size([4, 1024, 2h, 2w])
            x[2] : torch.Size([4, 2048,  h,  w])
        '''           
            
        x = self.encoder(x)        
        # print(f"\t(after RTDETR.encoder)")
        # for i in range(len(x)):
        #     print(f"\tx[{i}] : {x[i].shape}")
        '''
            x[0] : torch.Size([4, 256, 4h, 4w])
            x[1] : torch.Size([4, 256, 2h, 2w])
            x[2] : torch.Size([4, 256,  h,  w])
            
            notation of hybrid_encoder.py :
                x = outs = [fusion_F5-S4-S3, fusion_F5-S4-S3-S4, fusion_F5-S4-S3-S4-F5]
                    fusion_F5-S4-S3.shape       = [b, 256, 4h, 4w]
                    fusion_F5-S4-S3-S4.shape    = [b, 256, 2h, 2w]
                    fusion_F5-S4-S3-S4-F5.shape = [b, 256,  h,  w]
        ''' 
        
        x = self.decoder(x, targets)
        '''
            print(f"(after RTDETR.decoder)")
            
            for key, value in x.items():
                if key == 'pred_logits':
                    print(f"\t{key} : {value.shape}")
                    --> # pred_logits : torch.Size([4, 300, 80])
                
                elif key == 'pred_boxes':
                    print(f"\t{key} : {value.shape}")
                    --> # pred_boxes : torch.Size([4, 300, 4])
                    
                elif key == 'aux_outputs' or key == 'dn_aux_outputs': # key : list(key, value)
                    print(f"\t{key} : ")
                    for i in range(len(value)):
                        for k, v in value[i].items():
                            print(f"\t\t{key}[{i}][{k}] : {v.shape}")
                            --> # aux_outputs : 
                                    # aux_outputs[0][pred_logits] : torch.Size([4, 300, 80])
                                    # aux_outputs[0][pred_boxes] : torch.Size([4, 300, 4])
                                    # ...
                                    # aux_outputs[5][pred_logits] : torch.Size([4, 300, 80])
                                    # aux_outputs[5][pred_boxes] : torch.Size([4, 300, 4])
                            --> # dn_aux_outputs : 
                                    # dn_aux_outputs[0][pred_logits] : torch.Size([4, 200, 80])
                                    # dn_aux_outputs[0][pred_boxes] : torch.Size([4, 200, 4])
                                    # ...
                                    # dn_aux_outputs[5][pred_logits] : torch.Size([4, 200, 80])
                                    # dn_aux_outputs[5][pred_boxes] : torch.Size([4, 200, 4])
                
                elif key == 'dn_meta' : # key : (key, value)
                    print(f"\t{key} : ")
                    for k, v in value.items():
                        print(f"\t\t{key}[{k}] : {v}")
                        --> # dn_meta : 
                                # dn_meta[dn_positive_idx] : tuple data type -> (tensor, tensor, tensor, tensor)
                                # dn_meta[dn_num_group] : scalar
                                # dn_meta[dn_num_split] : [scalar, scalar]
        '''
                
        
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
