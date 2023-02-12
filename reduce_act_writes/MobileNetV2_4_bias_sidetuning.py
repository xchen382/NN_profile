import torch.nn as nn
import torch
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup


        if expand_ratio == 1:
            self.conv = [nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        # nn.ReLU6(inplace=True),
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(oup)]
            self.conv = nn.Sequential(*self.conv)
        

            # self.conv = nn.Sequential(
            #     # dw
            #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU6(inplace=True),
            #     # pw-linear
            #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(oup),
            # )
        else:

            self.conv = [
                        # pw
                        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        # nn.ReLU6(inplace=True),
                        # dw
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        # nn.ReLU6(inplace=True),
                        # pw-linear
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(oup)]
            self.conv = nn.Sequential(*self.conv)


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # self.features = [conv_bn(3, input_channel, 2)]
        self.features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(input_channel)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features += [nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.last_channel)]


        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    net = mobilenet_v2(False)
    x = torch.empty(1,3,224,224)
    # x = torch.empty(1,3,256,256)
    layer = 0
    weight = []
    weight_side = []
    act = []

    read_forward = []
    write_forward = []
    memory_forward = []

    read_backward = []
    write_backward = []
    memory_backward = []


    # specific: record block start to indicate the side tuning.
    block_start_layer = []
    # features
    for block_idx,block in enumerate(net.features):
        try:
            # record start of a block
            block_start_layer.append(len(weight)-1)
            # size of weight_size
            weight_side.append(4*(x.size(1)*(3*3+1)+x.size(1)*(x.size(1)+1)))

            for subblock_idx, subblock in enumerate(block.conv):
                x=subblock(x)
                print('block-'+str(block_idx),'subblock-'+str(subblock_idx),subblock,x.size())
                layer += 1
                weight.append(4*sum([para.nelement() for para in subblock.parameters()]))
                print(4*sum([para.nelement() for para in subblock.parameters()]))
                act.append(x.nelement()*4)
        except:
            #specific: not a residual block
            block_start_layer.pop()
            weight_side.pop()

            x=block(x)
            print('block-'+str(block_idx),block,x.size())
            act.append(x.nelement()*4)
            # weight.append(subblock.input_channels*out_channels*kernel_size**2)
            weight.append(4*sum([para.nelement() for para in block.parameters()]))
            print(4*sum([para.nelement() for para in block.parameters()]))

        # print('block-'+str(block_idx),'parameters(B):'+str((4*x.nelement())))
    # classifier
    x = x.mean(3).mean(2)
    x=net.classifier(x)
    weight.append(4*sum([para.nelement() for para in net.classifier.parameters()]))
    act.append(x.nelement()*4)
    print('classifier',net.classifier,4*sum([para.nelement() for para in net.classifier.parameters()]))
    print(4*(net.classifier.in_features *net.classifier.out_features))



    batch_size = 64

    # compute forward step
    read_forward = []
    write_forward = []
    memory_forward = [sum(weight)] # base memory is all the weights
    #specific: record the memory, read, and write for side network.
    act = [i*batch_size for i in act]

    # specific: check layer-wise memory to decide which to update

    print('layer-wise memory',[i+j for i,j in zip(act,weight)])

# specific: update which layer's weight and bias. 
    # weight updating: forward-store the input of the layer; backward-read gradient of output, read weight, read activations; write input gradient and weight gradient
    # weight updating: forward-store nothing; backward-read gradient of output, read weight; write input gradient
    layer_update_weight = []
    layer_update_bias = list(range(len(weight)))

    for layer_index, new_act in enumerate(act):
        if layer_index in layer_update_weight:
            memory_forward.append(memory_forward[-1]+new_act)
        else:
            memory_forward.append(memory_forward[-1]+0)
        #specific: save low act. for side tuning of 2 conv
        if layer_index in block_start_layer:
             memory_forward[-1] += new_act/4*2


    side_index = 0
    for layer_index in range(len(act)):
        new_read = [weight[layer_index],act[layer_index]]
        # read input, read weight
        read_forward.append(sum(new_read))
        if layer_index in block_start_layer:
            read_forward.append(weight_side[side_index])
            side_index += 1


    for layer_index,new_write in enumerate(act):
        # write output
        if layer_index in layer_update_weight:
            write_forward.append(new_write)
        else:
            write_forward.append(0)
        if layer_index in block_start_layer:
            write_forward.append(new_write/4*2)


    # compute backward step
    read_backward = []
    write_backward = []
    memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
    for layer_index in range(len(act))[-1::-1]:
        used_act = act[layer_index]
        if layer_index in layer_update_weight:
            memory_backward.append(memory_backward[-1]-used_act)
        else:
            memory_backward.append(memory_backward[-1]-0)
        #specific: save low act. for side tuning of 2 conv
        if layer_index in block_start_layer:
             memory_backward[-1] -= new_act/4*2


    side_index = len(weight_side)-1
    for layer_index in range(len(act))[-1::-1]:
        new_read = [act[layer_index],weight[layer_index],act[layer_index-1]]
        if layer_index in layer_update_weight:
            # read gradient of output, weight, and input
            read_backward.append(sum(new_read))
        else:
            read_backward.append(new_read[0]+new_read[1])
        # read input of 2, weight of 2 and gradient of output of 2.
        if layer_index in block_start_layer:
            read_backward.append(act[layer_index]/4*2+weight_side[side_index]*2)
            side_index -= 1



    side_index = len(weight_side)-1

    for layer_index in range(len(act))[-1::-1]:
        new_write = [act[layer_index],weight[layer_index]]
        if layer_index in layer_update_weight:
            # write gradient of weight, and gradient of input
            write_backward.append(sum(new_write))
        elif layer_index in layer_update_bias:
            write_backward.append(new_write[0])
        else:
            write_backward.append(0)
        
        # write gradient of weights, gradient of input
        if layer_index in block_start_layer:
            write_backward.append(act[layer_index]/4*2+weight_side[side_index])
            side_index -= 1

        


    # print(memory_backward)
    # print(read_backward)
    # print(memory_backward)




    print(memory_forward+memory_backward,max(memory_forward+memory_backward))
    print(read_forward+read_backward,sum(read_forward+read_backward))
    print(write_forward+write_backward,sum(write_forward+write_backward))
    




