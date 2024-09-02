import numpy as np
import struct
import torch

def load_images(file_name):
    f=open(file_name,"rb")
    data=f.read()
    _,num,rows,cols=struct.unpack_from('>4I',data,0)
    total_images=num*rows*cols
    image=struct.unpack_from('>'+str(total_images)+'B',data,struct.calcsize('>4I'))
    a=np.array(image)
    images=a.reshape((num,rows,cols))
    images = 2*(images.astype(np.float32) / 255.0)-1
    return images

def load_labels(file_name):
    f=open(file_name,"rb")
    data=f.read()
    _,num=struct.unpack_from('>2I',data,0)
    label=struct.unpack_from('>'+str(num)+'B',data,struct.calcsize('>2I'))
    a=np.array(label)
    labels=a.reshape((num))
    return labels

train_images=load_images('./number_data/train-images.idx3-ubyte')
train_labels=load_labels('./number_data/train-labels.idx1-ubyte')
imgs=torch.from_numpy(train_images)
imgs=imgs.unsqueeze(1)
torch.save(imgs,'./number_data/images.pt')
labels=torch.from_numpy(train_labels)
torch.save(labels,'./number_data/labels.pt')