import os
import os.path as osp

images = "fet"
imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
print(imlist)