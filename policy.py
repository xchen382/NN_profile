
# from utils.find_closest_value import find_closest_value
def find_closest_value(A, C):
    if len(A) == 0:
        return 0
    sorted_A = sorted(A)[-1::-1]
    for value in sorted_A:
        if value<=C:
            return value
    

class policy:
    def __init__(self, weight, act, inp, flops, block_start_layer, weight_side, flops_side, layer_pw, batch_size):
        self.weight = weight 
        self.act = act
        self.inp = inp
        self.flops = flops
        self.block_start_layer = block_start_layer
        self.weight_side = weight_side
        self.flops_side = flops_side
        self.batch_size = batch_size
        self.layer_pw = layer_pw

    def baseline(self):
        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = self.batch_size
        

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]
        for new_act in act:
            memory_forward.append(memory_forward[-1]+new_act)
        for new_read in zip(weight,inp):
            # read input, read weight
            read_forward.append(sum(new_read))
        for new_write in act:
            # write output
            write_forward.append(new_write)
        for new_comp in flops:
            flops_forward.append(new_comp)

        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for used_act in act[-1::-1]:
            memory_backward.append(memory_backward[-1]-used_act)

        for new_read in zip(act[-1::-1],act[-1::-1],weight[-1::-1],inp[-1::-1]):
            # read gradient of output, weight, and input
            read_backward.append(sum(new_read))

        for new_write in zip(inp[-1::-1],weight[-1::-1]):
            # write gradient of weight, and gradient of input
            write_backward.append(sum(new_write))

        for new_comp in flops[-1::-1]:
            flops_backward.append(2*new_comp)

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))


        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def act_quant_8bit(self):
        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = self.batch_size

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size/4 for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size/4 for i in inp]
        weight = [i/4 for i in weight]

        for new_act in act:
            memory_forward.append(memory_forward[-1]+new_act)
        for new_read in zip(weight,inp,act):
            # read weight, read quat input, orginal output
            read_forward.append(new_read[0]+new_read[1]+new_read[2]*4)
        for new_write in act:
            # write quant output, write output
            write_forward.append(new_write*4)
            write_forward.append(new_write)
        for new_comp in flops:
            flops_forward.append(new_comp)


        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for used_act in act[-1::-1]:
            memory_backward.append(memory_backward[-1]-used_act)

        for new_read in zip(act[-1::-1],act[-1::-1],weight[-1::-1],inp[-1::-1]):
            # read quat gradient of output, weight, and quant input
            # read orginal gradient of input and gradient of weight to quantize
            read_backward.append(sum(new_read)+(new_read[2]+new_read[3])*4)

        for new_write in zip(inp[-1::-1],weight[-1::-1]):
            # write origin/quant. gradient of weight, and gradient of input
            write_backward.append(sum(new_write))
            write_backward.append(sum(new_write)*4)

        for new_comp in flops[-1::-1]:
            flops_backward.append(2*new_comp)

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))


        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def batch_size_1(self):
        weight = self.weight
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = 1
        

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]
        for new_act in act:
            memory_forward.append(memory_forward[-1]+new_act)
        for new_read in zip(weight,inp):
            # read input, read weight
            read_forward.append(sum(new_read))
        for new_write in act:
            # write output
            write_forward.append(new_write)
        for new_comp in flops:
            flops_forward.append(new_comp)

        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for used_act in act[-1::-1]:
            memory_backward.append(memory_backward[-1]-used_act)

        for new_read in zip(act[-1::-1],act[-1::-1],weight[-1::-1],inp[-1::-1]):
            # read gradient of output, weight, and input
            read_backward.append(sum(new_read))

        for new_write in zip(inp[-1::-1],weight[-1::-1]):
            # write gradient of weight, and gradient of input
            write_backward.append(sum(new_write))

        for new_comp in flops[-1::-1]:
            flops_backward.append(2*new_comp)

        read_forward = [i*self.batch_size for i in read_forward]
        read_backward = [i*self.batch_size for i in read_backward]
        
        write_forward = [i*self.batch_size for i in write_forward]
        write_backward = [i*self.batch_size for i in write_backward]

        flops_forward = [i*self.batch_size for i in flops_forward]
        flops_backward = [i*self.batch_size for i in flops_backward]


        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))


        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def sparse_layer(self):
        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = self.batch_size
        

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]

        # specific: update which layer's weight and bias. 
        # weight updating: forward-store the input of the layer; backward-read gradient of output, read weight, read activations; write input gradient and weight gradient
        # weight updating: forward-store nothing; backward-read gradient of output, read weight; write input gradient
        layer_update_weight = [46,50,58,64,70,76,88,94,len(weight)-2,len(weight)-1]
        # only the last linear layer has bias, hence the weight error has to be computed to the minimal weight
        layer_input_gradient = list(range(min(layer_update_weight),len(weight)))



        for layer_index, new_act in enumerate(act):
            if layer_index in layer_update_weight:
                memory_forward.append(memory_forward[-1]+new_act)
            else:
                memory_forward.append(memory_forward[-1]+0)
        for new_read in zip(weight,inp):
            # read input, read weight
            read_forward.append(sum(new_read))
        for layer_index,new_write in enumerate(act):
            # write output
            write_forward.append(new_write)

        for new_comp in flops:
            flops_forward.append(new_comp)

        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for layer_index in range(len(act))[-1::-1]:
            used_act = act[layer_index]
            if layer_index in layer_update_weight:
                memory_backward.append(memory_backward[-1]-used_act)
            else:
                memory_backward.append(memory_backward[-1]-0)


        for layer_index in range(len(act))[-1::-1]:
            new_read = [act[layer_index],act[layer_index],weight[layer_index],inp[layer_index]]
            if layer_index in layer_update_weight:
                # read gradient of output, weight, and input
                read_backward.append(sum(new_read))
            else:
                read_backward.append(new_read[0]+new_read[2])

        for layer_index in range(len(inp))[-1::-1]:
            new_write = [inp[layer_index],weight[layer_index]]
            if layer_index in layer_update_weight:
                # write gradient of weight, and gradient of input
                write_backward.append(sum(new_write))
            elif layer_index in layer_input_gradient:
                write_backward.append(new_write[0])
            else:
                # only when the bias updating is in consecutive layers.
                write_backward.append(0)

        for layer_index in range(len(flops))[-1::-1]:
            new_comp = flops[layer_index]
            if layer_index in layer_update_weight:
                flops_backward.append(2*new_comp)
            elif layer_index in layer_input_gradient:
                flops_backward.append(new_comp)
            else:
                flops_backward.append(0)

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))


        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def bias_sidetuning(self):
        # Note the side channel consists of 2 conv layers
        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        block_start_layer = self.block_start_layer
        weight_side = self.weight_side
        flops_side = self.flops_side 
        batch_size = self.batch_size

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)+sum(weight_side)] # base memory is all the weights and side weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]
        flops_side = [i*batch_size for i in flops_side]


        # specific: update which layer's weight and bias. 
        # weight updating: forward-store the input of the layer; backward-read gradient of output, read weight, read activations; write input gradient and weight gradient
        # weight updating: forward-store nothing; backward-read gradient of output, read weight; write input gradient
        layer_update_weight = []
        layer_input_gradient = list(range(2,len(weight)))

        for layer_index, new_act in enumerate(act):
            if layer_index in layer_update_weight:
                memory_forward.append(memory_forward[-1]+new_act)
            else:
                memory_forward.append(memory_forward[-1]+0)
            #specific: save low act. for side tuning of 2 conv
            if layer_index in block_start_layer:
                memory_forward[-1] += new_act/4*2


        side_index = 0
        for layer_index in range(len(weight)):
            new_read = [weight[layer_index],inp[layer_index]]
            # read input, read weight
            read_forward.append(sum(new_read))
            if layer_index in block_start_layer:
                # read downsampled input twice, read weight
                read_forward.append(weight_side[side_index]+inp[layer_index]/4*2)
                side_index += 1


        for layer_index,new_write in enumerate(act):
            # write output
            write_forward.append(new_write)

            if layer_index in block_start_layer:
                # write intermidiate activations of two conv layers. The size is equal to input
                write_forward.append(inp[layer_index]/4*2)
        
        side_index = 0
        for layer_index,new_comp in enumerate(flops):
            flops_forward.append(new_comp)
            # flops introduced by side-tunning
            if layer_index in block_start_layer:
                flops_forward[-1] += flops_side[side_index]
                side_index += 1



        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
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
            new_read = [act[layer_index],act[layer_index],weight[layer_index],inp[layer_index]]
            if layer_index in layer_update_weight:
                # read gradient of output, weight, and input
                read_backward.append(sum(new_read))
            else:
                read_backward.append(new_read[0]+new_read[2])
            # read input of 2, weight of 2 and gradient of output of 2.
            if layer_index in block_start_layer:
                # update side channel needs to read
                new_read_side_channel_1x1 = inp[layer_index]/4 + inp[layer_index]/4 + weight_side[side_index] + inp[layer_index]/4
                new_read_side_channel_3x3 = inp[layer_index]/4 + inp[layer_index]/4 + inp[layer_index]/4
                read_backward.append(new_read_side_channel_1x1+new_read_side_channel_3x3)
                side_index -= 1



        side_index = len(weight_side)-1
        for layer_index in range(len(act))[-1::-1]:
            new_write = [inp[layer_index],weight[layer_index]]
            if layer_index in layer_update_weight:
                # write gradient of weight, and gradient of input
                write_backward.append(sum(new_write))
            elif layer_index in layer_input_gradient:
                write_backward.append(new_write[0])
            else:
                write_backward.append(0)
            
            # write gradient of weights, gradient of input
            if layer_index in block_start_layer:
                new_write_side_channel_1x1 = weight_side[side_index] + inp[layer_index]/4
                new_write_side_channel_3x3 = inp[layer_index]/4
                write_backward.append(new_write_side_channel_1x1+new_write_side_channel_3x3)
                side_index -= 1

        side_index = len(weight_side)-1
        for layer_index in range(len(flops))[-1::-1]:
            new_comp = flops[layer_index]
            if layer_index in layer_update_weight:
                flops_backward.append(2*new_comp)
            elif layer_index in layer_input_gradient:
                flops_backward.append(new_comp)
            else:
                flops_backward.append(0)
            # flops introduced by side-tunning
            if layer_index in block_start_layer:
                flops_forward[-1] += 2*flops_side[side_index]
                side_index -= 1

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))
    
        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def reversible(self):
        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        block_end_layer = self.block_start_layer
        batch_size = self.batch_size

        

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]

        # actually we need to store act from first 2 and last layers, but we ignore it here.
        for layer_idx, new_act in enumerate(act):
            if layer_idx != max(block_end_layer):
                memory_forward.append(memory_forward[-1]+0)
            else:
                memory_forward.append(memory_forward[-1]+new_act)

        for new_read in zip(weight,act):
            # read input, read weight
            read_forward.append(sum(new_read))
        for new_write in act:
            # write output
            write_forward.append(new_write)
        for new_comp in flops:
            flops_forward.append(new_comp)


        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for layer_index in range(len(act))[-1::-1]:
            if layer_index in block_end_layer:
                new_act = sum(act[block_end_layer[block_end_layer.index(layer_index)-1]:layer_index])

                memory_backward.append(memory_backward[-1]+new_act-act[layer_index])
                # print([round(x) for x in act[block_end_layer[block_end_layer.index(layer_index)-1]:layer_index]],layer_index,memory_backward[-1])


            elif layer_index<max(block_end_layer):
                memory_backward.append(memory_backward[-1]-act[layer_index])
                # print(act[layer_index],layer_index,memory_backward[-1])

        for new_read in zip(act[-1::-1],act[-1::-1],weight[-1::-1],inp[-1::-1]):
            # the activations and weight are read once again to compute
            read_backward.append(new_read[2]+new_read[3])
            # read gradient of output, weight, and input
            read_backward.append(sum(new_read))



        for new_write in zip(act[-1::-1],inp[-1::-1],weight[-1::-1]):
            # write gradient of weight, and gradient of input, also recompute the activations
            write_backward.append(sum(new_write))

        for new_comp in flops[-1::-1]:
            flops_backward.append(3*new_comp)

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))
    
        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def sparsegrad(self):
        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = self.batch_size
        

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]

        weight_updating_reduction = 0.1

        for new_act in act:
            memory_forward.append(memory_forward[-1]+new_act)
        for new_read in zip(weight,inp):
            # read input, read weight
            read_forward.append(sum(new_read))
        for new_write in act:
            # write output
            write_forward.append(new_write)
        for new_comp in flops:
            flops_forward.append(new_comp)


        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for used_act in act[-1::-1]:
            memory_backward.append(memory_backward[-1]-used_act)

        for new_read in zip(act[-1::-1],act[-1::-1],weight[-1::-1],inp[-1::-1]):
            # read gradient of output, weight, and input
            read_backward.append(sum(new_read))

        for new_write in zip(inp[-1::-1],weight[-1::-1]):
            # write gradient of weight, and gradient of input
            
            write_backward.append(new_write[0]+weight_updating_reduction*new_write[1])

        for new_comp in flops[-1::-1]:
            flops_backward.append(2*new_comp)

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))

        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def randomfilter(self):

        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = self.batch_size

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]

        # specific: update which layer's weight and bias. 
        # weight updating: forward-store the input of the layer; backward-read gradient of output, read weight, read activations; write input gradient and weight gradient
        # weight updating: forward-store nothing; backward-read gradient of output, read weight; write input gradient
        layer_update_weight = self.layer_pw
        # only the last linear layer has bias, hence the weight error has to be computed to the minimal weight
        layer_input_gradient = list(range(min(layer_update_weight),len(weight)))



        for layer_index, new_act in enumerate(act):
            if layer_index in layer_update_weight:
                memory_forward.append(memory_forward[-1]+new_act)
            else:
                memory_forward.append(memory_forward[-1]+0)
        for new_read in zip(weight,inp):
            # read input, read weight
            read_forward.append(sum(new_read))
        for layer_index,new_write in enumerate(act):
            # write output
            write_forward.append(new_write)

        for new_comp in flops:
            flops_forward.append(new_comp)

        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for layer_index in range(len(act))[-1::-1]:
            used_act = act[layer_index]
            if layer_index in layer_update_weight:
                memory_backward.append(memory_backward[-1]-used_act)
            else:
                memory_backward.append(memory_backward[-1]-0)



        for layer_index in range(len(act))[-1::-1]:
            new_read = [act[layer_index],act[layer_index],weight[layer_index],inp[layer_index]]
            if layer_index in layer_update_weight:
                # read gradient of output, weight, and input
                read_backward.append(sum(new_read))
            else:
                read_backward.append(new_read[0]+new_read[2])

        for layer_index in range(len(inp))[-1::-1]:
            new_write = [inp[layer_index],weight[layer_index]]
            if layer_index in layer_update_weight:
                # write gradient of weight, and gradient of input
                write_backward.append(sum(new_write))
            elif layer_index in layer_input_gradient:
                write_backward.append(new_write[0])
            else:
                # only when the bias updating is in consecutive layers.
                write_backward.append(0)

        for layer_index in range(len(flops))[-1::-1]:
            new_comp = flops[layer_index]
            if layer_index in layer_update_weight:
                flops_backward.append(2*new_comp)
            elif layer_index in layer_input_gradient:
                flops_backward.append(new_comp)
            else:
                flops_backward.append(0)

        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))


        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]


    def recompute(self):


        weight = self.weight  
        act = self.act
        inp = self.inp
        flops = self.flops
        batch_size = self.batch_size
        

        # layer_store_act = list(range(200))
        layer_store_act = []

        # compute forward step
        read_forward = []
        write_forward = []
        flops_forward = []
        memory_forward = [sum(weight)] # base memory is all the weights
        act = [i*batch_size for i in act]
        flops = [i*batch_size for i in flops]
        inp = [i*batch_size for i in inp]
        for layer_idx, new_act in enumerate(act):
            if layer_idx in layer_store_act:
                memory_forward.append(memory_forward[-1]+new_act)
            else:
                memory_forward.append(memory_forward[-1]+0)

        for new_read in zip(weight,inp):
            # read input, read weight
            read_forward.append(sum(new_read))
        for new_write in act:
            # write output
            write_forward.append(new_write)
        for new_comp in flops:
            flops_forward.append(new_comp)

        memory_forward = memory_forward[1::]
        # compute backward step
        read_backward = []
        write_backward = []
        flops_backward = []
        memory_backward = [memory_forward[-1]] # base memory is from all the activations are stored
        for layer_index in range(len(act))[-1::-1]:
            used_act = act[layer_index]
            if layer_index in layer_store_act:
                memory_backward.append(memory_backward[-1]-used_act)
            else:
                memory_backward.append(memory_backward[-1]-0)

        for layer_index in range(len(act))[-1::-1]:
            new_read = [act[layer_index],act[layer_index],weight[layer_index],inp[layer_index]]
            # read gradient of output, weight, and input
            read_backward.append(sum(new_read))
            if not(layer_index in layer_store_act):
                start_layer = find_closest_value(layer_store_act,layer_index)
                recompute_new_read = [weight[recompute_layer]+inp[recompute_layer] for recompute_layer in range(start_layer,layer_index)]
                read_backward[-1] += sum(recompute_new_read)
                    
        for layer_index in range(len(inp))[-1::-1]:        
            new_write = [inp[layer_index],weight[layer_index]]
            # write gradient of weight, and gradient of input
            write_backward.append(sum(new_write))
            if not(layer_index in layer_store_act):
                start_layer = find_closest_value(layer_store_act,layer_index)
                recompute_new_write = [act[recompute_layer] for recompute_layer in range(start_layer,layer_index)]
                write_backward[-1] += sum(recompute_new_write)


        for layer_index in range(len(flops))[-1::-1]:
            new_comp = flops[layer_index]
            flops_backward.append(2*new_comp)
            if not(layer_index in layer_store_act):
                start_layer = find_closest_value(layer_store_act,layer_index)
                recompute_new_flops= [flops[recompute_layer] for recompute_layer in range(start_layer,layer_index)]
                flops_backward[-1] += sum(recompute_new_flops)


        print('read(B):',sum(read_forward+read_backward))
        print('write(B):',sum(write_forward+write_backward))
        print('FLOPs(B):',sum(flops_forward+flops_backward))
        print('Peakmem(B):',max(memory_forward+memory_backward))


        return read_forward,write_forward,flops_forward,memory_forward,read_backward[-1::-1],write_backward[-1::-1],flops_backward[-1::-1],memory_backward[-1::-1]







