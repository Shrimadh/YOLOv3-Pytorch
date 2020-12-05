import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes):
    batch_size = prediction.size(0)        #  While testing it is 1
    stride = inp_dim // prediction.size(2) #  416//(13 or 26 or 52) which gives strides of 46 23 11
    grid_size = inp_dim // stride          #  416//(46 or 23 or 11) which gives grid_size
    bbox_attrs = 5 + num_classes           #  5 + 80
    num_anchors = len(anchors)             #  3

    # Reshapes precition[1,255,13,13] to [1,255,169] 
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)

    # Reshapes prediction[1,255,169] to [1,169,255]
    prediction = prediction.transpose(1,2).contiguous()
    
    # Reshapes prediction[1,169,255] to [1,169*3 ,85]
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Rescales anchor boxes according to the stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    # Gets the x,y co-ordinate and the objectness score to a value between 0 and 1 using sigmoid
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Arranges numbers till grid_size in a vector [ 0 1 2 3 4 5 6 7 8 9 10 11 12] for grid_size = 13
    grid = np.arange(grid_size)

    # Makes a grid where 
    # a =  0 1 . . 12
    #      .
    #      .
    #      0 1 . . 12
    # b =  0 0 . . 0
     #     1 1 . . 1  
    #      .
    #      .
    #      12 12 . . 12  
    a,b = np.meshgrid(grid, grid)

    # Reshapes a and b to have one row and the other dimension is automatically adjusted
    x_offset = torch.FloatTensor(a).view(-1,1)  # [169,1]
    y_offset = torch.FloatTensor(b).view(-1,1)  # [169,1]
    
    # Concatenates x and y offset and tiles it and reshapes it such that last dim is 2
    x_y_offset = torch.cat((x_offset,y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)  # [169,2] to [507,2] to [1,507,2]
    prediction[:,:,:2] += x_y_offset     
                                                             
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)  # [1,507,2]

    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors  
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    prediction[:,:,:4] *= stride
    return prediction

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #Co-orrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # Create a confidence mask and multiply the whole predictions output with it
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) 
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    

    # We go through NMS for each image in the batch seperately
    for ind in range(batch_size):
        image_pred = prediction[ind] #Takes one image from each row
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4],as_tuple = False))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2],as_tuple = False).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > threshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4],as_tuple = False).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0

def load_classes(namesfile):
    fp = open(namesfile,"r")
    names = fp.read().split("\n")[:-1]
    return names

def letterbox_image(img,inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w,h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas

def prep_image(img, inp_dim):

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img















"""a = torch.rand(4,10647,85)
b = a[1]
conf_mask = (a[:,:,4] > 0.8).float().unsqueeze(2)
print(conf_mask.size())
a = a*conf_mask
print(a)
max_conf, max_conf_scores = torch.max(b[:,5:5+ 80], 1)
print(max_conf,max_conf_scores)
print(max_conf.size(),max_conf_scores.size())

max_conf = max_conf.float().unsqueeze(1)
print(max_conf.size(),max_conf_scores.size())

max_conf_scores = max_conf_scores.float().unsqueeze(1)
print(max_conf,max_conf_scores)

seq = (b[:,:5], max_conf, max_conf_scores)
image_pred = torch.cat(seq, 1)
print(image_pred.size())
print(image_pred[:,-1])
print(image_pred[:,-2])

non_zero_ind =  torch.nonzero(image_pred[:,4],as_tuple = False)
print(non_zero_ind.size())
print(image_pred.size())
print(image_pred[:,-1])
print(image_pred[:,-2])
print(non_zero_ind.squeeze())
try:
    image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
except:
    print("ok")
print(image_pred.size())
img_classes = unique(image_pred_[:,-1])
print(img_classes)
cls_mask=image_pred_*(image_pred_[:,-1] == 1).float().unsqueeze(1)
class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
print(class_mask_ind,class_mask_ind.size())
print(image_pred_[:,-1])
print(image_pred_[:,-2])
image_pred_class = image_pred_[class_mask_ind].view(-1,7)


conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
print(torch.sort(image_pred_class[:,4], descending = True ))
print(conf_sort_index)
image_pred_class = image_pred_class[conf_sort_index]
idx = image_pred_class.size(0)
print(image_pred_class.size())
"""
