import torch.nn as nn
import torch
import math
from fvcore.nn import FlopCountAnalysis
from utils.write_to_csv import write_scv
import argparse
from transformers import BertTokenizer, BertModel

from model_analysis import model_analysis_bert
from policy import policy
from utils.write_to_csv import write_scv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read/Write/Peakmemory for Bert')
    parser.add_argument('--policy', default='6_sparsegrad', type=str, help='Policy to reduce the peakmemory')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--sequence_length', default=512, type=int, help='Batch size')
    args = parser.parse_args()

    net = BertModel.from_pretrained("bert-base-uncased")
    text_input = "dummy"
    for _ in range(args.sequence_length - 1):
        text_input += " dummy"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # x = tokenizer(text_input, return_tensors='pt')
    x = torch.empty(1, args.sequence_length,768)
    print(net)
    [module_name, weight, act, inp, flops, 
    block_start_layer, weight_side, flops_side,
                                    layer_pw] = model_analysis_bert(x,net)

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
