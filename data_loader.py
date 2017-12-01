import torch
import torch.utils.data as utils_data
import numpy as np
import torchvision.transforms as transforms
import os
import scipy.io
from scipy.misc import imread

path='/Users/shreesh/Academics/CS670/Project/'
data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

def getLR(imageFileName):
    return data_transforms(imread(path+'LR/'+imageFileName)).numpy()

def getHR(imageFileName):
    return data_transforms(imread(path+'HR/'+imageFileName)).numpy()

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

def data_loader(batchSize, threads):

    filenames = labels()
    images_lr = []
    images_hr = []
    for file in filenames:
        if 'png' not in file:continue
        images_lr.append(getLR(file))
        images_hr.append(getHR(file))
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

def data_loader_transfer():
    data = {}
    numClasses = 103
    for i in range(1, numClasses):
        data[i] = []
    images = os.listdir(path+'/LR')
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

    j = 0
    for i in labels:
        if 'png' in images[j] : data[i].append(images[j])
        j += 1

    dataset = []
    for i in data:
        cur_image_names = data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]
        dataset.append([getLR(data[i][0]), getHR(data[i][0]), [getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][1])]])
        dataset.append([getLR(data[i][1]), getHR(data[i][1]), [getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][0])]])
        dataset.append([getLR(data[i][2]), getHR(data[i][2]), [getHR(data[i][1]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][0])]])
        dataset.append([getLR(data[i][3]), getHR(data[i][3]), [getHR(data[i][2]), getHR(data[i][0]), getHR(data[i][4]), getHR(data[i][1])]])
        dataset.append([getLR(data[i][4]), getHR(data[i][4]), [getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][0]), getHR(data[i][1])]])
    
    return dataset

def data_loader_transfer_test():
    data = {}
    numClasses = 103
    for i in range(1, numClasses):
        data[i] = []
    images = os.listdir(path+'/LR')
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

    j = 0
    for i in labels:
        if 'png' in images[j] : data[i].append(images[j])
        j += 1

    dataset = []
    for i in data:
        dataset.append([getLR(data[i][0]), getHR(data[i][1]), getHR(data[i][2])])
        if(len(dataset)==10): break
    
    return dataset