#-*-coding:utf8-*-

import cv2
import torch
import numpy as np

from dataset import *
from model import *
from datetime import datetime


def now():
	return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

n = 2
dataset = TestDataset("./dataset/images/test")

print("%s | load model..." % now())
model = torch.load("checkpoint/epoch_28.pth", map_location='cpu')
model = model.module
print("%s | finish" % now())

#if torch.cuda.is_available():
#	model = model.cuda()

print("%s | get batch" % now())
data = dataset.data()
print("%s | generating..." % now())

for idx, item in enumerate(data):
	output = model(data=item, get_output=True)
	
	_, preds = torch.max(output, dim=1)
	pred = preds[0].numpy()
	pred[pred == 1] = 255
	pred = pred[:,:,np.newaxis]
	pred = pred.repeat([3], axis=2)
	
	source = item["source"].numpy()[0]
	source = source.transpose((1, 2, 0))
	
	output = np.hstack((source, pred))
	
	cv2.imwrite("output/%d.png" % idx, output)

print("%s | finish" % now())
