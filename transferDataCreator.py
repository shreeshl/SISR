import os
from skimage.transform import downscale_local_mean
import numpy as np
path = '/Users/sreekar/Downloads/jpg/'
data = {}
numClasses = 103
for i in range(1, numClasses):
	data[i] = []
images = os.listdir(path)
labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]


j = 0

def getLR(imageFileName):
	return np.array([0,0,2])

def getHR(imageFileName):
	return np.array([0,0,3])
dataset = {}
for i in labels:
	data[i].append(images[j])
	j += 1

for i in data:
	cur_image_names = data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]
	dataset[i] = [getLR(data[i][0]), getHR(data[i][1]), getHR(data[i][2]), getHR(data[i][3]), getHR(data[i][4]), getHR(data[i][0])]



def getLR(imageFileName):
	return np.array([0,0,2])

def getHR(imageFileName):
	return np.array([0,0,3])