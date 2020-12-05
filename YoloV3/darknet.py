# File to construct the darknet-53

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import * 

# Function returns a list of blocks
# by parsing through the cfg file 

def parse_cfg(cfgfile):

    file = open(cfgfile,'r')
    lines = file.read().split('\n')               # Lines is a list
    lines = [x for x in lines if len(x) > 0]      # No need to read empty lines
    lines = [x for x in lines if x[0] != '#']     # Remove comments
    lines = [x.rstrip().lstrip() for x in lines]  # Remove Whitespace
    
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0] # Hyperparameters info
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index,x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if(x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            # Adds a conv layer to the sequential module
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Adds a BatchNorm layer to the sequential module
            bn = nn.BatchNorm2d(filters)
            module.add_module("batch_norm_{0}".format(index), bn)

            # Adds an activation layer
            if activation == "leaky":
                acti = nn.LeakyReLU(0.1,inplace = True)
                module.add_module("leaky_{0}".format(index), acti)
        
        # Upsample layer     
        elif(x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2 , mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        
        # Route layer
        elif(x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            # Empty layer or dummy layer in the module_list
            # During forward pass concatenation will be done
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)
            if end < 0:
                filters = output_filters[start + index] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        # Dummy layer again
        # Filters will be added in forward pass
        elif(x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)
        
        elif(x["type"] == "yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            # The lowest scales have the biggest boxes
            # So the first prediction with 13 cells detects the big objects and the anchor boxes are suited for that\
            # There are totally 9 anchor boxes in all

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [(anchors[i]) for i in mask]

            #Detection Layer is sort of a dummy layers but in addition it holds the anchor boxes shape
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i,module in enumerate(modules):
            mt = (module["type"])

            if(mt == "convolutional" or mt == "upsample"):
                x = self.module_list[i](x) 
            
            elif mt == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    # Feature map concatenation
                    x = torch.cat((map1, map2), 1)

            elif  mt == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
            
            elif mt == 'yolo':        

                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int (self.net_info["height"])

                # Get the number of classes
                num_classes = int (module["classes"])

                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:              # If no collector has been intialised. 
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections
    
    def load_weights(self,weightfile):
        fp = open(weightfile,"rb")

        # 1.Major version
        # 2.Minor version
        # 3.Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32 , count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp,dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")




