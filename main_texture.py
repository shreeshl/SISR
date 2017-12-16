import argparse, os
import torch
import torchvision.transforms as transforms
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet import *
import data_loader
import numpy as np
from torchvision import models
from srresnet import GramMatrix
import torch.utils.model_zoo as model_zoo

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=15, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=50, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--transfer", action="store_true", help="Use content loss?")

def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)    

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    # dataset = data_loader.data_loader_transfer()
    dataset = data_loader.data_loader(opt.batchSize, 4)

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(size_average=False)

    if cuda:
        print("===> Setting GPU")
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda() 

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    
    # ignored_params = list(map(id, model.features.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,model.parameters())
    print("===> Setting Optimizer")
    # for p in model.features.parameters():
    #     p.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # optimizer = optim.Adam([{'params': base_params,'lr': opt.lr},
                            # {'params': model.features.parameters()}], 
                            # lr=opt.lr*0.1, momentum=0.9)

    # optimizer = optim.Adam(base_params,lr=opt.lr*0.1, momentum=0.9)
    f = open("training_details_%s"%(time.strftime("%m_%d-%H_%M")),"w")
    
    vgg = VGG()
    vgg.load_state_dict(torch.load('vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()
    
    
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        
        t = time.time()
        lr = adjust_learning_rate(optimizer, epoch-1)
        total_loss = 0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr  

        print "epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]
        model.train()

        for iteration, batch in enumerate(dataset, 1):

            # input, target, transfer = get_batch(dataset, iteration, opt.batchSize)
            # input, target, transfer = Variable(torch.from_numpy(input)), Variable(torch.from_numpy(target)), Variable(torch.from_numpy(transfer))
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
                # transfer = transfer.cuda()

            style_layers = ['r11','r21','r31'] 
            
            weights = [1e2/n**2 for n in [64,128,256]]
            # weights = [3e-7, 1e-6, 1e-6]

            
            loss = []
            output = model(input)
            loss.append(criterion(output, target))
            gram_target  = [GramMatrix()(A).detach() for A in vgg(target, style_layers)]
            gram_output  = [GramMatrix()(A).detach() for A in vgg(output, style_layers)]
            index = 0
            for m1, m2 in zip(gram_output, gram_target):
                loss.append(weights[index] * criterion(m1,m2))
                index+=1
            # for m1, m2 in zip(gram_target, gram_output):
            #     loss.append(weights[index] * criterion(m1,m2))
            #     index+=1
            # for i in range(0,140,20):
            #     for j in range(0,140,20):
            #         gram_transfer  = [GramMatrix()(A).detach() for A in vgg(transfer[:,:,i:i+20,j:j+20], style_layers)]
            #         input_transfer = [GramMatrix()(A).detach() for A in vgg(output[:,:,i:i+20,j:j+20], style_layers)]
            #         index = 0
            #         for m1, m2 in zip(gram_transfer, input_transfer):
            #             loss.append(weights[index] * criterion(m1,m2))
            #             index+=1

            final_loss = sum(loss)
            optimizer.zero_grad()
            
            final_loss.backward()

            optimizer.step()
            total_loss+=final_loss.data[0]
            
            if iteration%opt.batchSize == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(dataset), final_loss.data[0]))
                f.write("===> Epoch[%d](%d/%d): Loss: {:%.3f}\n"%(epoch, iteration, len(dataset), final_loss.data[0]))
        
        print("Avg Loss : %.2f"%(total_loss/iteration))
        f.write("Epoch %d Avg Loss : %.2f\n"%(epoch, total_loss/iteration))
        f.flush()

        print("Time taken in seconds : %.2f",time.time()-t)
        save_checkpoint(model, epoch)
    f.close()
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def get_batch(dataset, iteration, bs):
    startidx = ((iteration-1)*bs)%len(dataset)
    endidx   = min(startidx+bs, len(dataset))
    input, target, transfer = [], [], []
    for idx in range(startidx,endidx):
        input.append(dataset[idx][0])
        target.append(dataset[idx][1])
        transfer.append(dataset[idx][2][random.randint(0,len(dataset[idx][2])-1)])
    
    return np.asarray(input), np.asarray(target), np.asarray(transfer)
    
def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
