import torch
import torch.utils.data as utils_data
import numpy as np
import torchvision.transforms as transforms
import os
import scipy.io
from scipy.misc import imread

path='/Users/shreesh/Academics/CS670/Project/'

def labels():
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]
    files  = os.listdir(path+'/HR')
    filenames = []
    i = 0
    while i<len(labels):
        curr_label = labels[i]
        if i+2<len(labels):
            filenames.append(files[i])
            filenames.append(files[i+1])
            filenames.append(files[i+2])
        while i<len(labels) and curr_label==int(labels[i]):
            i+=1
    return filenames

def labels_transfer():
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]
    files  = os.listdir(path+'/HR')
    filenames = []
    i = 0
    while i<len(labels):
        curr_label = labels[i]
        if i+2<len(labels):
            filenames.append(files[i])
            filenames.append(files[i+1])
            filenames.append(files[i+2])
        while i<len(labels) and curr_label==int(labels[i]):
            i+=1
    return filenames

def data_loader(batchSize, threads):

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    filenames = labels()
    images_lr = []
    images_hr = []
    for file in filenames:
        if 'png' not in file:continue
        lr = data_transforms(imread(path+'LR/'+file)).numpy()
        hr = data_transforms(imread(path+'HR/'+file)).numpy()
        images_lr.append(lr)
        images_hr.append(hr)
        if(len(images_lr)==15): break

    images_lr = np.asarray(images_lr)
    images_hr = np.asarray(images_hr)

    images_lr = torch.from_numpy((images_lr))
    images_hr = torch.from_numpy((images_hr))

    training_samples = utils_data.TensorDataset(images_lr, images_hr)
    data_loader = utils_data.DataLoader(training_samples, batch_size=batchSize, shuffle=True, num_workers = threads)

    return data_loader

def data_loader_test():

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    filenames = labels()
    np.random.seed(1)
    idxs = np.arange(0, len(filenames))
    np.random.shuffle(idxs)
    images_lr = []
    images_hr = []
    for idx in idxs[:10]:
        file = filenames[idx]
        if 'png' not in file:continue
        lr = data_transforms(imread(path+'LR/'+file)).numpy()
        hr = data_transforms(imread(path+'HR/'+file)).numpy()
        images_lr.append(lr)
        images_hr.append(hr)
    
    return [images_lr, images_hr]



# ymenoptera_dataset2 = datasets.ImageFolder(path,data_transforms['train'])
# training_samples = utils_data.TensorDataset(ymenoptera_dataset2, ymenoptera_dataset2)
# dataset_loader = torch.utils.data.DataLoader(transform,batch_size=4, shuffle=True,num_workers=4)