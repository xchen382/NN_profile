import torch.nn as nn
import torch
import math
from fvcore.nn import FlopCountAnalysis
from utils.write_to_csv import write_scv
import argparse
from model import mobilenet_v2
from model_analysis import model_analysis
from policy import policy
from utils.write_to_csv import write_scv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read/Write/Peakmemory for MobileNetV2')
    parser.add_argument('--policy', default='baseline', type=str, help='Policy to reduce the peakmemory')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    args = parser.parse_args()

    net = mobilenet_v2(False)

    x = torch.empty(1,3,224,224)

    [module_name, weight, act, inp, flops, 
    block_start_layer, weight_side, flops_side,
                                    layer_pw] = model_analysis(x,net)

    P = policy(weight, act, inp, flops, block_start_layer, weight_side, flops_side, layer_pw, args.batch_size)
    
    if args.policy == 'baseline':
        result = P.baseline()
    elif args.policy == '1_act_quant':
        result = P.act_quant_8bit()
    elif args.policy == '2_batchsize1_gradacc':
        result = P.batch_size_1()
    elif args.policy == '3_sparse_layer':
        result = P.sparse_layer()
    elif args.policy == '4_biase_sidetuning':
        result = P.bias_sidetuning()
    elif args.policy == '5_reversible':
        result = P.reversible()
    elif args.policy == '6_sparsegrad':
        result = P.sparsegrad()
    elif args.policy == '7_randomfilter':
        result = P.randomfilter()
    elif args.policy == '9_recompute':
        result = P.recompute()
    else:
        raise ValueError('The current policy is not supported yet.')



    [read_forward,write_forward,flops_forward,memory_forward,
    read_backward,write_backward,flops_backward,memory_backward] = result

    write_scv(args.policy, module_name,weight,inp,act,
             read_forward,write_forward,flops_forward,memory_forward,
             read_backward,write_backward,flops_backward,memory_backward)
