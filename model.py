#-*-coding:utf8-*-

import torch
import torchvision
import numpy as np
import torch.nn as nn

from torchvision import models
from collections import OrderedDict


class MyModel(nn.Module):

	def __init__(self, num_classes):
		super().__init__()

		self.__num_classes = num_classes
		self.__crit = nn.NLLLoss()
	
		self.__encoder = nn.Sequential(OrderedDict([
			("conv1", nn.Conv2d(3, 16, kernel_size=(3, 3))),
			("norm1", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu1", nn.ReLU()),
			("conv2", nn.Conv2d(16, 16, kernel_size=(3, 3))),
			("norm2", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu2", nn.ReLU()),
			("conv3", nn.Conv2d(16, 16, kernel_size=(3, 3))),
			("norm3", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu3", nn.ReLU())]))
		self.__decoder = nn.Sequential(OrderedDict([
			("transposed_conv1", nn.ConvTranspose2d(16, 16, kernel_size=(3, 3))),
			("norm4", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu4", nn.ReLU()),
			("transposed_conv2", nn.ConvTranspose2d(16, 16, kernel_size=(3, 3))),
			("norm5", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu5", nn.ReLU()),
			("transposed_conv3", nn.ConvTranspose2d(16, num_classes, kernel_size=(3, 3))),
			("norm6", nn.BatchNorm2d(num_classes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu6", nn.ReLU()),
			("log_softmax", nn.LogSoftmax())]))	

	def forward(self, data, get_output=False):
		x = self.__encoder(data["source"])
		x = self.__decoder(x)

		if not get_output:
			loss = self.__crit(x, data["label"])
			return loss
		else:
			return x


class MyModel2(nn.Module):

	def __init__(self, num_classes):
		super().__init__()

		self.__num_classes = num_classes
		self.__crit = nn.NLLLoss()
	
		self.__encoder = nn.Sequential(OrderedDict([
			("conv1", nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1, dilation=1)),
			("norm1", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu1", nn.ReLU()),
			("conv2", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=2, dilation=2)),
			("norm2", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu2", nn.ReLU()),
			("conv3", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=3, dilation=3)),
			("norm3", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu3", nn.ReLU()),
			("conv4", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, dilation=1)),
			("norm4", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu4", nn.ReLU()),
			("conv5", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=2, dilation=2)),
			("norm5", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu5", nn.ReLU()),
			("conv6", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=3, dilation=3)),
			("norm6", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu6", nn.ReLU()),
			("conv7", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, dilation=1)),
			("norm7", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu7", nn.ReLU()),
			("conv8", nn.Conv2d(16, 16, kernel_size=(3, 3), padding=2, dilation=2)),
			("norm8", nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu8", nn.ReLU()),
			("conv9", nn.Conv2d(16, num_classes, kernel_size=(3, 3), padding=3, dilation=3)),
			("norm9", nn.BatchNorm2d(num_classes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
			("relu9", nn.ReLU()),
			("log_softmax", nn.LogSoftmax())]))

	def forward(self, data, get_output=False):
		x = self.__encoder(data["source"])

		if not get_output:
			loss = self.__crit(x, data["label"])
			return loss
		else:
			return x
