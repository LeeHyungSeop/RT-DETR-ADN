'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# 2024.05.23 @hslee : RepBlock for feature fusion in CCFF block
class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        
        residual1 = src
        
        if self.normalize_before:
            src = self.norm1(src)
        
        '''
        print(f"\t\t\t\t[self_att]")
        '''
        q = k = self.with_pos_embed(src, pos_embed)
        '''
        print(f"\t\t\t\t\tQuery : {q.shape}")
        print(f"\t\t\t\t\tKey : {k.shape}")
        print(f"\t\t\t\t\tValue : {src.shape}")
        '''
        
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = self.dropout1(src)
        src = residual1 + src
        '''
        print(f"\t\t\t\tself_attn output : {src.shape}")
        '''
        
        if not self.normalize_before:
            src = self.norm1(src)

        residual2 = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.dropout(self.activation(self.linear1(src)))
        src = self.linear2(self.dropout2(src))
        '''
        print(f"\t\t\t\tlinear1, linear2 output : {src.shape}")
        '''
        
        src = residual2 + src
        if not self.normalize_before:
            src = self.norm2(src)
        
        '''
            src.shape=torch.Size([b, hw, 256])
            q.shape = k.shape = torch.Size([b, hw, 256])
            
            forward : 
                residual1=src [b, hw, 256]
                src -> q,k=pos_embed(src) -> self_attn(q, k, src)-> dropout1(src) -> src [b, hw, 256]
                src+=residual1 -> src=norm1(src) [b, hw, 256]
                
                residual2=src [b, hw, 256]
                src -> linear1(256, 1024) -> activation -> dropout -> src[b, hw, 1024]
                src -> linear2(1024, 256) ->               dropout -> src[b, hw,  256]
                src+=residual2 -> src=norm2(src) [b, hw, 256]
        '''
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for idx, layer in enumerate(self.layers):
            '''
            print(f"\t\t[encoder layer {idx} input] : {output.shape}")
            '''
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
            '''
            print(f"\t\t[encoder layer {idx} output] : {output.shape}")
            '''
            

        if self.norm is not None:
            output = self.norm(output)
        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 in_channels_swinT=[192, 384, 768],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 backbone='resnet',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # 2024.05.23 @hslee : make all the input channels to 256(hidden_dim)
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:   # in_channels=[512, 1024, 2048]
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        # 2024.06.15 @hslee : make all the input channels to 256(hidden_dim) for SwinTransformer
        self.input_proj_swinT = nn.ModuleList()
        for in_channel in in_channels_swinT:   # in_channels_swinT=[192, 384, 768]
            self.input_proj_swinT.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
            
            
        

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        # 2024.05.23 @hslee
        '''
        print(f"num_encoder_layers={num_encoder_layers}")
        print(f"len(use_encoder_idx)={len(use_encoder_idx)}")
        '''
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # hidden_dim = 256
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, kernel_size=1, stride=1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, kernel_size=3, stride=2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        
        # 2024.05.23 @hslee : make all the input channels to 256(hidden_dim)
        
        # for SwinTransformer
        if feats[0].shape[1] == 192:
            proj_feats = [self.input_proj_swinT[i](feat) for i, feat in enumerate(feats)]
        # for ResNet variants
        else :
            proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        ''' 
            (feat_high)
            proj_feats[2] S5 : torch.Size([b, 2048,  h,  w]) --> torch.Size([b, 256,  h,  w])
            proj_feats[1] S4 : torch.Size([b, 1024, 2h, 2w]) --> torch.Size([b, 256, 2h, 2w])
            proj_feats[0] S3 : torch.Size([b,  512, 4h, 4w]) --> torch.Size([b, 256, 4h, 4w])
            (feat_low) 
        '''
        
        # HybridEncoder start
        '''
        print("[HybridEncoder] : ")
        '''
        if self.num_encoder_layers > 0:
            # use_encoder_idx=[2] -> so only S5([S3, S3, S5]) is used for encoder
            # only i=0, enc_ind=2 because use_encoder_idx=[2]
            for i, enc_ind in enumerate(self.use_encoder_idx):   
                h, w = proj_feats[enc_ind].shape[2:] # (b, c, h, w)
                
                # flatten [b, c, h, w] to [b, hw, c]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1) 
                
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                # S5[b, hw, c] -> self.encoder + pos_embed -> F5[b, hw, c]
                '''
                print(f"\t[AIFI (Attention-based Intra-scale Feature Interaction)] : encoder(S5) -> F5")
                print(f"\t\tencoder input = S5 : {src_flatten.shape}")
                '''
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                '''
                print(f"\t\tencoder output = F5(flatten) : {memory.shape}") 
                '''
                
                # F5[b, hw, c] -> F5[b, c, h, w]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous() 
                '''
                print(f"\t\tencoder output = F5(unflatten) : {proj_feats[enc_ind].shape}", end='\n\n')
                '''
                
                # print([x.is_contiguous() for x in proj_feats ])
        
        
        # 2024.05.23 @hslee : CCFF(CNN-based Cross-scale Feature Fusion)
        # broadcasting and fusion
        ''' 
            (feat_high)
            proj_feats[2]  S5 -> encoder -> F5 : torch.Size([b, 256,  h,  w])
            proj_feats[1]                   S4 : torch.Size([b, 256, 2h, 2w])
            proj_feats[0]                   S3 : torch.Size([b, 256, 4h, 4w])
            (feat_low)
        '''
        F5 = proj_feats[-1]
        inner_outs = [F5]
        '''
        print(f"\t[CCFF (Convolution-based Cross Feature Fusion)] : fusion(F5, S4, S3)")
        print(f"\t\tF5 : {F5.shape}")
        print(f"\t\tS4 : {proj_feats[1].shape}")
        print(f"\t\tS3 : {proj_feats[0].shape}")
        '''
        
        for idx in range(len(self.in_channels) - 1, 0, -1):   # in_channels=[512, 1024, 2048]
            # 0
            feat_high = inner_outs[0]       
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high) # same shape
            feat_low = proj_feats[idx - 1]  
            inner_outs[0] = feat_high
            # 1
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            # 2
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            # 3
            inner_outs.insert(0, inner_out)
            '''
                inner_outs = [F5]
                
                idx2 
                    # 0) (feat_high, feat_low) = (F5, S4)
                    # 1) F5.shape = [b, 256, h, w] -> (upsample) -> [b, 256, 2h, 2w]
                    # 2) concat(F5, S4) -> [b, 512, 2h, 2w] -> (CSPRepLayer) -> fusion_F5-S4.shape[b, 256, 2h, 2w])
                    # 3) inner_outs = [fusion_F5-S4, F5]
                idx1
                    # 0) (feat_high, feat_low) = (fusion_F5-S4, S3)
                    # 1) fusion_F5-S4.shape = [b, 256, 2h, 2w] -> (unsample) -> [b, 256, 4h, 4w]
                    # 2) concat(fusion_F5-S4, S3) -> [b, 512, 4h, 4w] -> (CSPRepLayer) -> fusion_F5-S4-S3.shape[b, 256, 4h, 4w])
                    # 3) inner_outs = [fusion_F5-S4-S3, fusion_F5-S4, F5]
                    
                inner_outs = [fusion_F5-S4-S3, fusion_F5-S4, F5]
            '''
        
        '''
        print(f"\t\t\tintermediate fusion output : [fusion_F5-S4-S3, fusion_F5-S4, F5]")
        print(f"\t\t\tintermediate fusion output : [{inner_outs[0].shape}, {inner_outs[1].shape}]")
        '''
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):  # in_channels=[512, 1024, 2048]
            # 0
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            # 1
            downsample_feat = self.downsample_convs[idx](feat_low)
            # 2
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            # 3
            outs.append(out)
            '''
                outs = [fusion_F5-S4-S3]
                
                idx 0
                    # 0) (feat_low, feat_high) = (fusion_F5-S4-S3, fusion_F5-S4)
                    # 1) fusion_F5-S4-S3.shape = [b, 256, 4h, 4w] -> (downsample) -> [b, 256, 2h, 2w]
                    # 2) concat(fusion_F5-S4-S3, fusion_F5-S4) -> [b, 512, 2h, 2w] -> (CSPRepLayer) -> fusion_F5-S4-S3-S4.shape[b, 256, 2h, 2w])
                    # 3) outs = [fusion_F5-S4-S3, fusion_F5-S4-S3-S4]
                idx 1
                    # 0) (feat_low, feat_high) = (fusion_F5-S4-S3-S4, F5)
                    # 1) fusion_F5-S4-S3-S4.shape = [b, 256, 2h, 2w] -> (downsample) -> [b, 256, h, w]
                    # 2) concat(fusion_F5-S4-S3-S4, F5) -> [b, 512, h, w] -> (CSPRepLayer) -> fusion_F5-S4-S3-S4-F5.shape[b, 256, h, w])
                    # 3) outs = [fusion_F5-S4-S3, fusion_F5-S4-S3-S4, fusion_F5-S4-S3-S4-F5]
                
                outs = [fusion_F5-S4-S3, fusion_F5-S4-S3-S4, fusion_F5-S4-S3-S4-F5]
                    fusion_F5-S4-S3.shape       = [b, 256, 4h, 4w]
                    fusion_F5-S4-S3-S4.shape    = [b, 256, 2h, 2w]
                    fusion_F5-S4-S3-S4-F5.shape = [b, 256, h, w]
            '''
            
        '''
        print(f"\t\t\tfinal fusion output : [fusion_F5-S4-S3, fusion_F5-S4-S3-S4, fusion_F5-S4-S3-S4-F5]")
        print(f"\t\t\tfinal fusion output : [{outs[0].shape}, {outs[1].shape}, {outs[2].shape}]")
        '''
        
        return outs
