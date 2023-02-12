import torch.nn as nn
import torch
import math
from fvcore.nn import FlopCountAnalysis
import argparse

def side_analysis(x,in_channel):
    side_net = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, groups=in_channel, bias=True),
                            nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True))
    x_downsample = torch.empty((x.size(0),x.size(1),x.size(2)//2,x.size(3)//2))
    side_flops = FlopCountAnalysis(side_net,x_downsample).total()
    side_weight = sum([4*para.nelement() for para in side_net.parameters()])
    print(side_flops,side_weight)
    return side_weight, side_flops

def model_analysis(x,net):

    module_name = []
    weight = []
    act = []
    inp = []
    flops = []

    block_start_layer = []
    weight_side = []
    flops_side = []

    layer_pw=[]




    # features
    for block_idx,block in enumerate(net.features):
        try:
            # record start of a block
            block_start_layer.append(len(weight))
            # size of weight_size
            side_para = side_analysis(x,x.size(1))
            weight_side.append(side_para[0])
            flops_side.append(side_para[1])

            for subblock_idx, subblock in enumerate(block.conv):
                module_name.append(str(subblock))
                flops.append(FlopCountAnalysis(subblock, x).total())
                inp.append(x.nelement()*4)
                x=subblock(x)
                print('block-'+str(block_idx),'subblock-'+str(subblock_idx),subblock,x.size())
                weight.append(4*sum([para.nelement() for para in subblock.parameters()]))
                print(4*sum([para.nelement() for para in subblock.parameters()]),flops[-1])
                act.append(x.nelement()*4)
                if isinstance(subblock,nn.Conv2d):
                    if subblock.kernel_size[0] == 1:
                        layer_pw.append(len(weight)-1)
        except:
            #specific: not a residual block
            block_start_layer.pop()
            weight_side.pop()
            flops_side.pop()
            

            module_name.append(str(block))
            flops.append(FlopCountAnalysis(block, x).total())
            inp.append(x.nelement()*4)
            x=block(x)
            print('block-'+str(block_idx),block,x.size())
            act.append(x.nelement()*4)
            # weight.append(subblock.input_channels*out_channels*kernel_size**2)
            weight.append(4*sum([para.nelement() for para in block.parameters()]))
            print(4*sum([para.nelement() for para in block.parameters()]),flops[-1])

        # print('block-'+str(block_idx),'parameters(B):'+str((4*x.nelement())))
    # classifier
    x = x.mean(3).mean(2)
    inp.append(x.nelement()*4)
    flops.append(FlopCountAnalysis(net.classifier, x).total())
    x=net.classifier(x)
    module_name.append(str(net.classifier))
    weight.append(4*sum([para.nelement() for para in net.classifier.parameters()]))
    act.append(x.nelement()*4)
    print('classifier',net.classifier,4*sum([para.nelement() for para in net.classifier.parameters()]))
    print(4*(net.classifier.in_features *net.classifier.out_features),flops[-1])

    return module_name, weight, act, inp, flops, block_start_layer, weight_side, flops_side, layer_pw

