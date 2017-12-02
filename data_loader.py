import torch
import torch.utils.data as utils_data
import numpy as np
import torchvision.transforms as transforms
import os
from skimage.transform import resize
import scipy.io
from scipy.misc import imread

path='/Users/shreesh/Academics/CS670/Project/'
LR = 'LR_small/'
HR = 'HR_small/'
data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

def getLR(imageFileName):
    return data_transforms(imread(path+LR+imageFileName)).numpy()

def getHR(imageFileName, imageFileName2 = 'None'):
    if imageFileName2!='None':
        new_transfer_im = descrambled_image(imread(path+LR+imageFileName2), imread(path+HR+imageFileName))
        # return np.expand_dims(new_transfer_im,axis=0)
        return np.transpose(new_transfer_im,(2,0,1)).astype('float32')
    return data_transforms(imread(path+HR+imageFileName)).numpy()

def labels():
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]
    files  = os.listdir(path+HR)
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
        lr = data_transforms(imread(path+LR+file)).numpy()
        hr = data_transforms(imread(path+HR+file)).numpy()
        images_lr.append(lr)
        images_hr.append(hr)
    
    return [images_lr, images_hr]

def data_loader_transfer():
    data = {}
    numClasses = 18
    for i in range(1, numClasses):
        data[i] = []
    images = os.listdir(path+LR)
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

    # j = 0
    # for i in labels:
    #     if 'png' in images[j] : data[i].append(images[j])
    #     j += 1

    for i in range(len(images)):
        if 'png' in images[i] : data[int(i/80)+1].append(images[i])
    
    dataset = []
    for i in data:
        
        # dataset.append([getLR(data[i][0]), getHR(data[i][0]), [getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][1])]])
        # dataset.append([getLR(data[i][1]), getHR(data[i][1]), [getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][0])]])
        # dataset.append([getLR(data[i][2]), getHR(data[i][2]), [getHR(data[i][1]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][0])]])
        # dataset.append([getLR(data[i][3]), getHR(data[i][3]), [getHR(data[i][2]), getHR(data[i][0]), getHR(data[i][4]), getHR(data[i][1])]])
        # dataset.append([getLR(data[i][4]), getHR(data[i][4]), [getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][0]), getHR(data[i][1])]])

        dataset.append([getLR(data[i][0]), getHR(data[i][0]), [getHR(data[i][2],data[i][0]), getHR(data[i][3],data[i][0])]])
        dataset.append([getLR(data[i][1]), getHR(data[i][1]), [getHR(data[i][2],data[i][1]), getHR(data[i][3],data[i][1])]])
        dataset.append([getLR(data[i][2]), getHR(data[i][2]), [getHR(data[i][1],data[i][2]), getHR(data[i][3],data[i][2])]])
        dataset.append([getLR(data[i][3]), getHR(data[i][3]), [getHR(data[i][2],data[i][3]), getHR(data[i][0],data[i][3])]])
        dataset.append([getLR(data[i][4]), getHR(data[i][4]), [getHR(data[i][2],data[i][4]), getHR(data[i][3],data[i][4])]])
        # if(len(dataset)==10): break
    
    return dataset

def data_loader_transfer_test():
    data = {}
    numClasses = 103
    for i in range(1, numClasses):
        data[i] = []
    images = os.listdir(path+LR)
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

    j = 0
    for i in labels:
        if 'png' in images[j] : data[i].append(images[j])
        j += 1

    dataset = []
    for i in data:
        dataset.append([getLR(data[i][0]), getHR(data[i][0]), getHR(data[i][2])])
        if(len(dataset)==10): break
    
    return dataset

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def descrambled_image(im1, im2):
    #im1 low  resolution image
    #im2 high resolution image
    
    im1 = resize(im1, (160,160))
    im2 = im2/255.0
    hr_size = im2.shape[0]
    new_hr  = np.zeros((hr_size,hr_size,3))
    cell_size = 40
    for i in range(0,hr_size, cell_size):
        for j in range(0, hr_size, cell_size):
            lr_patch = im1[i:i+cell_size,j:j+cell_size,:]
            k_max = 0
            l_max = 0
            min_error = 10**6
            for k in range(0, hr_size, cell_size):
                for l in range(0, hr_size, cell_size):
                    hr_patch = im2[k:k+cell_size,l:l+cell_size,:]
                    res = hr_patch-lr_patch
                    error = np.sum(np.abs(res))
                    if error<min_error:
                        min_error = error
                        k_max = k
                        l_max = l
            new_hr[i:i+cell_size,j:j+cell_size,:] = im2[k_max:k_max+cell_size, l_max:l_max+cell_size, :]
    
    return new_hr








