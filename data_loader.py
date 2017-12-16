import torch
import torch.utils.data as utils_data
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
from skimage.color import rgb2gray
from torch.autograd import Variable
from scipy.misc import imresize
from imageio import imresize
import time
from torchvision import models

# path='/home/shreesh/pytorch-SRResNet-master/'
path='/Users/shreesh/Academics/CS670/Project/'
LR = 'LR_small/'
HR = 'HR_small/'
data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
vgg = models.vgg16_bn(pretrained=True)
# model = nn.Sequential(*list(vgg.features.children()))

def getLR(imageFileName):
    return data_transforms(imread(path+LR+imageFileName)).numpy()

def getHR(imageFileName, imageFileName2 = 'None'):
    if imageFileName2!='None':
        new_transfer_im = descrambled_image(imread(path+LR+imageFileName2), imread(path+HR+imageFileName))
        # return np.expand_dims(new_transfer_im,axis=0).astype('float32')
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
    filenames = os.listdir(path+'VGG_small')
    np.random.seed(1)
    idxs = np.arange(0, len(filenames))
    np.random.shuffle(idxs)
    images_lr = []
    images_hr = []
    for idx in idxs[:10]:
        file = filenames[idx]
        if notValidFile(file):continue
        lr = data_transforms(imread(path+LR+file[:-3]+'png')).numpy()
        hr = data_transforms(imread(path+HR+file[:-3]+'png')).numpy()
        images_lr.append(lr)
        images_hr.append(hr)
    
    return [images_lr, images_hr]

def notValidFile(filename):
    if 'png' not in filename and 'jpg' not in filename: return 1
    return 0

def data_loader_transfer():
    data = {}
    numClasses = 103
    for i in range(1, numClasses):
        data[i] = []
    images = os.listdir(path+LR)
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]
    images.sort()

    j = 0
    # for i in labels:
    #     if notValidFile(images[i]): continue
    #     data[i].append(images[j])
    #     j += 1
    
    for i in range(len(images)):
        if notValidFile(images[i]): continue
        index = int(images[i][:-4].split("_")[-1])
        data[int(index/81)+1].append(images[i][:-3]+'png')
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
    numClasses = 18
    for i in range(1, numClasses):
        data[i] = []
    images = os.listdir(path+LR)
    labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]
    images.sort()
    # j = 0
    # for i in labels:
    #     if 'png' in images[j] : data[i].append(images[j])
    #     j += 1
    for i in range(len(images)):
        if 'png' in images[i] : data[int(i/80)+1].append(images[i])

    dataset = []
    
    for idx,i in enumerate(data):
        # dataset.append([getLR(data[i][0]), getHR(data[i][0]), getHR(data[i][2])])
        t = time.time()
        dataset.append([getLR(data[i][0]), getHR(data[i][0]), getHR(data[i][2],data[i][0])])
        dataset.append([getLR(data[i][5]), getHR(data[i][5]), getHR(data[i][4],data[i][5])])
        print time.time()-t, idx
        if(idx==13): break
        
    
    return dataset

def descrambled_image(im1, im2):
    #im1 low  resolution image
    #im2 high resolution image
    mean = np.asarray([0.45795686, 0.48501961, 0.40760392]).reshape(1,1,3)
    std = np.asarray([0.224, 0.225, 0.229]).reshape(1,1,3)
    im1 = resize(im1, (160,160))
    im2 = im2/255.0
    hr_size = im2.shape[0]
    new_hr  = np.zeros((hr_size,hr_size,3))
    cell_size = 40
    # im1_features = {}
    # im2_features = {}
    # t = time.time()
    # for i in range(0,hr_size, cell_size):
    #     for j in range(0, hr_size, cell_size):
    #         im1_patch = im1[i:i+cell_size,j:j+cell_size] 
    #         im2_patch = im2[i:i+cell_size,j:j+cell_size] 
    #         im1_patch = np.expand_dims(im1_patch.transpose(2,0,1),0)
    #         im1_patch = Variable(torch.from_numpy(im1_patch).float())
    #         im2_patch = np.expand_dims(im2_patch.transpose(2,0,1),0)
    #         im2_patch = Variable(torch.from_numpy(im2_patch).float())
    #         im1_features[i,j] = extract_hypercolumn(vgg, [2,5], im1_patch)
    #         im2_features[i,j] = extract_hypercolumn(vgg, [2,5], im2_patch)
    
    # for i in range(0,hr_size, cell_size):
    #     for j in range(0, hr_size, cell_size):
    #         if(hr_size-i<cell_size or hr_size-j<cell_size): continue
    #         im2_patch = im2[i:i+cell_size,j:j+cell_size]
    #         im2_patch = np.expand_dims(im2_patch.transpose(2,0,1),0)
    #         im2_patch = Variable(torch.from_numpy(im2_patch).float())
    #         im2_features[i,j] = model(im2_patch).data[0].numpy()[:,0,]

    im1_hyper = extract_hypercolumn(vgg, [3,8,15,22,29], Variable(torch.from_numpy(im1).view(3,160,160).unsqueeze(0).float()))
    im2_hyper = extract_hypercolumn(vgg, [3,8,15,22,29], Variable(torch.from_numpy(im2).view(3,160,160).unsqueeze(0).float()))

    # weights = np.dot(im1_hyper.transpose(1,2,0).reshape(-1,1216), im2_hyper.reshape(1216,-1))
    # new_hr = np.dot(softmax(weights),im2.reshape(-1,3))
    t=time.time()
    m = nn.Softmax(dim=0)
    for i in range(0,hr_size, cell_size):
        for j in range(0, hr_size, cell_size):
            lr_patch = im1[i:i+cell_size,j:j+cell_size,:]
            # lr_patch = im1_features[i,j]
            k_max = 0
            l_max = 0
            min_error = 10**6
            
            for k in range(0, hr_size, 10):
                for l in range(0, hr_size, 10):
                    if(hr_size-k<cell_size or hr_size-l<cell_size): continue
                    hr_patch = im2[k:k+cell_size,l:l+cell_size,:]
                    # hr_patch = im2_features[k,l]
                    res = hr_patch-lr_patch
                    # error = np.sum(np.abs(res))
                    error = np.sum(res**2)
                    if error<min_error:
                        min_error = error
                        k_max = k
                        l_max = l
            weights = np.dot(im1_hyper[:,i:i+cell_size,j:j+cell_size].reshape(im1_hyper.shape[0],-1).T,\
                            im2_hyper[:,k_max:k_max+cell_size, l_max:l_max+cell_size].reshape(im1_hyper.shape[0],-1))
            weights = m(Variable(torch.from_numpy(weights))).data.numpy()
            weights[weights<0.1]=0
            new_hr[i:i+cell_size,j:j+cell_size,:] = np.dot(weights,im2[k_max:k_max+cell_size, l_max:l_max+cell_size, :].reshape(-1,3)).reshape(cell_size,cell_size,3)

            # print min_error
    print time.time()-t
    
    # return rgb2gray(new_hr)
    return new_hr


def cosine(vecx, vecy):
    norm = np.sqrt(np.dot(vecx, vecx))* np.sqrt(np.dot(vecy, vecy))
    return np.dot(vecx, vecy) / (norm + 1e-10)

def extract_hypercolumn(model, layer_indexes, image):
    """
    Returns hypercolumns of size(___x40x40) when you pass input image of (3,40,40).
    layer_indexes is list of layers of whose features_maps we want to add. We can send a list of all layers or any 
    specific layers we want. For eg below, I sent [2,5].
    """

    layers = [nn.Sequential(*list(model.features.children())[:l]) for l in layer_indexes]
    features_maps = [feats(image) for feats in layers]
    hypercolumns = []
    for convmap in features_maps:
        cmap = convmap.squeeze(0)
        for fmap in cmap:
            fmap = fmap.data.numpy()
            upscaled = imresize(fmap, size=(image.size()[2],image.size()[2]), mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

def diff_to_be_added(model, vgg, lr, t_hr):
    """
    lr is (3,40,40). t_hr is (3,60,60)(whatever, doesn't matter).
    t_hr_lr is reconstructed image of hr(see the commented line-you HAVE to uncomment). So it is supposed to be same size as lr. This is most imp.
    I repeat: t_hr_lr and lr should have same exact size.
    """

    lr = lr.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    lr = model(Variable(torch.from_numpy(lr).float()))
    LR_hypercolumns = extract_hypercolumn(vgg, [2,5], lr)
    t_hr_lr = resize(t_hr.transpose(1,2,0),(40,40))
    t_hr_lr = t_hr_lr.reshape(1,t_hr_lr.shape[2],t_hr_lr.shape[0],t_hr_lr.shape[1])
    t_hr_lr = model(Variable(torch.from_numpy(t_hr_lr).float()))
    t_hr_lr_data = t_hr_lr.data.numpy()[0]
    t_hr_lr_descrambled = descrambled_image(t_hr_lr_data, lr.data.numpy()[0])
    HR_hypercolumns = extract_hypercolumn(vgg, [2,5], t_hr_lr_descrambled)
    

    to_be_added = np.zeros((3,160,160))

    for i in range(160):
        for j in range(160):
            LR_hypercolumn_pixel = LR_hypercolumns[:,i,j]
            nearest_sim = -100
            nearest_neighbor = (i,j)
            for m in range(10):
                for n in range(10):
                    HR_hypercolumn_pixel = HR_hypercolumns[:,m,n]
                    sim = cosine(HR_hypercolumn_pixel, LR_hypercolumn_pixel)
                    if sim > nearest_sim:
                        nearest_sim = sim
                        nearest_neighbor = (m,n)
            m,n = nearest_neighbor
            to_be_added[:,i,j] = t_hr[:,m,n] - t_hr_lr_descrambled[:,m,n]
            print i, j
    return to_be_added





