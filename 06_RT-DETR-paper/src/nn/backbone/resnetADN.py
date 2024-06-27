from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import misc as misc_nn_ops

from ._api import  WeightsEnum
from ._utils import _ovewrite_named_param
from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = [
    "ResNetADN",
    "resnet50",
    "resnet101",
]

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}

donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            # 2024.06.04 @hslee : set norm_layer to FrozenBatchNorm2d
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        skippable: bool = False, # is this block skippable? @woochul
    ) -> None:
        super().__init__()

        self.skippable = skippable

        if norm_layer is None:
            # 2024.06.04 @hslee : set norm_layer to FrozenBatchNorm2d
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.skippable == False:   # Switch BN for shared layers.  @woochul
            self.bn1_skip = norm_layer(width)
            self.bn2_skip = norm_layer(width)
            self.bn3_skip = norm_layer(planes * self.expansion)
        
            
        # 2024.04.08 @hslee
        # for test purpose, 
        # "To test whether BN and BN_Skip are appropriately utilized."
        # self.bn1_cnt = 0
        # self.bn1_skip_cnt = 0

    def forward(self, x: Tensor, skip: bool = False) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.skippable == False and skip == True:  # Switchable BN. @woochul
            # print(f"bn1_skip is used")
            out = self.bn1_skip(out)
        else:
            # print(f"bn1 is used")
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.skippable == False and skip == True:  # Switchable BN. @woochul
            # print(f"bn2_skip is used")
            out = self.bn2_skip(out)
        else:
            # print(f"bn2 is used")
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.skippable == False and skip == True: # Switchable BN. @woochul
            # print(f"bn3_skip is used")
            out = self.bn3_skip(out)
        else:
            # print(f"bn3 is used")
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SkippableSequentialBlocks(nn.Sequential):
    """Skips some blocks in the stage"""
    def forward(self, input, skip = False):
        """Extends nn.Sequential's forward for skipping some blocks
        Args:
            x (Tensor): input tensor
            skip (bool): if True, skip the last half blocks in the stage.
        """
        for i in range(len(self)):
            if self[i].skippable == True and skip == True:
                # print(f"--skip {type(self[i])}")
                pass
            else:
                # print(f"--execute {type(self[i])}")
                input = self[i](input, skip)
        # print("")
        return input

@register
class ResNetADN(nn.Module):
    def __init__(
        self,
        depth,
        block: Type[Union[Bottleneck,]],
        groups: int = 1,
        width_per_group: int = 64,
        # 2024.06.08 @hslee : add parameters
        num_stages=4, 
        freeze_at=-1, 
        freeze_norm=True, 
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        pretrained=False) : 
        super().__init__()

        # 2024.06.03 @hslee
        block_nums = ResNet_cfg[depth]
        self.num_skippable_stages = len(block_nums)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.block_expansion = 4
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        # 2024.06.03 @hslee : make layers skippable
        for i, num_blocks in enumerate(block_nums):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            planes = 64 * 2**i
            dilate = False
            stride = 1
            blocks = block_nums[i]
            # First half blocks are shared and not skippable. @woochul
            # 1. default: n_shared == n_skippable
            n_shared = (blocks + 1) // 2  

            print(f"make layer i: {i}, num_blocks: {num_blocks}, num_shared: {n_shared}, num_skippable: {blocks - n_shared}")
                        
            if i != 0:
                dilate = replace_stride_with_dilation[i - 1]
                stride = 2
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * self.block_expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * self.block_expansion, stride),
                    norm_layer(planes * self.block_expansion),
                )

            block = Bottleneck if depth >= 50 else BasicBlock
            
            block_layers = []
            block_layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, skippable=False
                )
            )
            self.inplanes = planes * self.block_expansion
            for i_block in range(1, blocks):
                block_layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                        skippable = (i_block >= n_shared), #last half blocks are skippable
                    )
                )
            setattr(self, f"layer{i+1}", SkippableSequentialBlocks(*block_layers))
        
        # 2024.06.03 @hslee
        # this part must be modified my resnet50_adn
        
        # pretrained parameter의 key와 model의 key가 다를 경우, 
        # strict=False로 설정하면, 같은 key 값만 matching되어 load되고, 나머지 key들은 random initialization된다.
        # -> base model의 baseline을 학습할 때 문제가 됨 (저자가 제공한 pretrained parameter의 key와 ResNet50ADN model의 key가 다름)
        
        
        # # 1 : my ResNet50ADN
        # if pretrained :
        #     path = "/home/hslee/Desktop/RetinaNet-ADN/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth"
        #     state = torch.load(path)['model']
            
        #     # 2024.06.04 @hslee
        #     # remove keys : "fc.weight", "fc.bias"
        #     # because full resnet50 architecture including fc.weight and fc.bias is not used in RT-DETR backbone
        #     del state["fc.weight"]
        #     del state["fc.bias"]
            
        #     # # print all parameter
        #     # for name, param in state.items():
        #     #     print(f"name : {name}, param : {param}")
            
        #     self.load_state_dict(state, strict=True)
        #     print(f"Load ResNet{depth}-ADN state_dict from {path} -----------------------------------------------")
            
            # print("here")
            # # print all parameter
            # for name, param in self.named_parameters():
            #     print(f"name : {name}, param : {param}")
            
        # # 2 : original ResNet50 (provided by paper's author)
        # if pretrained:
        #     state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            
        #     # print all parameters
        #     for name, param in state.items():
        #         print(f"name : {name}, param : {param}")
            
        #     self.load_state_dict(state, strict=True   )
        #     print(f"Load PResNet{depth} state_dict -----------------------------------------------")
            
        #     # print all parameter
        #     for name, param in self.named_parameters():
        #         print(f"name : {name}, param : {param}")
        
        # # 3 : PyTorch ResNetV1 pretrained weight
        if pretrained:
            # load pytorch resnet50v1 pretrained model (acc@1 : 76.130, acc@5 : 92.862)
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            
            # load pytorch resnet50v2 pretrained model (acc@1 : 80.858, acc@5 : 95.434)
            # url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            
                
            state = torch.hub.load_state_dict_from_url(url)
            
            del state["fc.weight"]
            del state["fc.bias"]
        
        self.load_state_dict(state, strict=False)
        print(f"Load state_dict from {url}")
        
        
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)
                
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m
            
    
    

    def _forward_impl(self, x: Tensor, skip: List[bool] = None) -> Tensor:
        # See note [TorchScript super()]

        outs = []
        
        assert self.num_skippable_stages == len(skip), \
            f"The networks has {self.num_stages} skippable stages, got: {len(skip)}"

        # 2024.03.22 @hslee
        # print("here is resnet.py > _forward_impl")
        # print(f"skip: {skip}")
        # print(f"x.shape: {x.shape}")
        
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x, skip[0])
        '''
        (8h, 8w, 256)
        '''
        
        x = self.layer2(x, skip[1])
        outs.append(x)
        '''
        (4h, 4w, 512)
        '''
        
        x = self.layer3(x, skip[2])
        outs.append(x)
        '''
        (2h, 2w, 1024)
        '''
        
        x = self.layer4(x, skip[3])
        outs.append(x)
        '''
        (h, w, 2048)
        '''

        # for i in range(len(outs)):
        #     print(f"\tout[{i}] : {outs[i].shape}")

        return outs

    def forward(self, x: Tensor, skip: List[bool] = None) -> Tensor:
        # skip = [False, False, False, False]
        # skip = [True, False, False, False]
        # skip = [False, True, False, False]
        # skip = [False, False, True, False]
        # skip = [False, False, False, True]
        # skip = [True, True, False, False]
        # skip = [True, False, True, False]
        # skip = [True, False, False, True]
        # skip = [False, True, True, False]
        # skip = [False, True, False, True]
        # skip = [False, False, True, True] 
        # skip = [True, True, True, False]
        # skip = [True, True, False, True]
        # skip = [True, False, True, True]
        # skip = [False, True, True, True]
        # skip = [True, True, True, True] 
        if skip is None:
            skip = [False for _ in range(self.num_skippable_stages)]
        return self._forward_impl(x, skip = skip)


def _resnet(
    block: Type[Union[Bottleneck,]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNetADN:
    if weights is not None:
        # _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        _ovewrite_named_param(kwargs, "num_classes", 1000)

    model = ResNetADN(block, layers, **kwargs)

    if weights is not None:
        # 2024.03.22 @hslee
        # remove key(s) in state_dict: "fc.weight", "fc.bias". in weights['model']
        # because full resnet50 architecture including fc.weight and fc.bias is not used in RetinaNet backbone.
        state_dict = weights['model']
        model.load_state_dict(weights['model'])
    return model

def resnet50(*, weights = None, progress: bool = True, test_only=False, **kwargs: Any) -> ResNetADN:    
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

def resnet101(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNetADN:
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
