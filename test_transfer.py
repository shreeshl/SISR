import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
import data_loader

parser = argparse.ArgumentParser(description="PyTorch SRResNet Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_10.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model,map_location=lambda storage, location: storage)["model"]
model2 = torch.load('../model_epoch_128.pth',map_location=lambda storage, location: storage)["model"]
model.eval()
model2.eval()
test_images = data_loader.data_loader_transfer_test()
for i in range(len(test_images)):
    im_b  = test_images[i][0]
    im_gt = test_images[i][1]
    im_transfer = test_images[i][2]
    # im_transfer = np.zeros((3,160,160))
    im_input = im_b
    im_input    = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    im_transfer = im_transfer.reshape(1,im_transfer.shape[0],im_transfer.shape[1],im_transfer.shape[2])
    im_input = Variable(torch.from_numpy(im_input).float())
    im_transfer = Variable(torch.from_numpy(im_transfer).float())

    # if cuda:
    #     model = model.cuda()
    #     im_input = im_input.cuda()
    #     im_transfer = im_transfer.cuda()

    model = model.cpu()
    model2 = model2.cpu()
        
    start_time = time.time()
    out = model(im_input, im_transfer)
    out2 = model2(im_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()
    out2 = out2.cpu()

    im_h = out.data[0].numpy().astype(np.float32)
    im_h2 = out2.data[0].numpy().astype(np.float32)

    im_h = im_h*255.
    im_h[im_h<0] = 0
    im_h[im_h>255.] = 255.            
    im_h = im_h.transpose(1,2,0)

    im_h2 = im_h2*255.
    im_h2[im_h2<0] = 0
    im_h2[im_h2>255.] = 255.            
    im_h2 = im_h2.transpose(1,2,0)

    print("It takes {}s for processing".format(elapsed_time))
    print("PSNR : %.2f",PSNR(im_h, im_gt.transpose(1,2,0)))

    fig = plt.figure(figsize=(12, 3))
    ax = plt.subplot("151")
    ax.imshow(im_gt.transpose(1,2,0))
    ax.set_title("GT")

    ax = plt.subplot("152")
    ax.imshow(test_images[i][2].transpose(1,2,0))
    ax.set_title("Transfer")

    ax = plt.subplot("153")
    ax.imshow(im_b.transpose(1,2,0))
    ax.set_title("Input(Bicubic)")

    ax = plt.subplot("154")
    # ax.imshow(rgb2gray(im_h.astype(np.uint8)), cmap = 'gray')
    ax.imshow(im_h.astype(np.uint8))
    ax.set_title("SRResNet_T")

    ax = plt.subplot("155")
    # ax.imshow(rgb2gray(im_h2.astype(np.uint8)), cmap = 'gray')
    ax.imshow(im_h2.astype(np.uint8))
    ax.set_title("SRResNet")
    plt.savefig("test_transfer/%d.png"%i)
    # plt.show()
