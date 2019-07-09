#-*-coding:utf8-*-

import os
import cv2
import torch
import random
import torchvision
import numpy as np

from torchvision import transforms


class DataRow:
	
	def __init__(self):
		self.source = None
		self.label = None
		self.category = ""


class DatasetBase:

	def __init__(self, path):
		self.__annotations_path = "%s/annotations" % path
		self.__images_path = "%s/images" % path
		self._data = []
		self.__width = 256
		self.__height = 256
		self._index = list(range(len(self._data)))

	def _get_data(self, type_):
		images_path = "%s/%s" % (self.__images_path, type_)
		label_path = "%s/%s" % (self.__annotations_path, type_)
		data = []

		cnt = 0		
		for file_ in os.listdir(images_path):
			file_ = file_[:file_.rfind('-')]

			if not os.path.exists("%s/%s" % (images_path, file_+"-org.jpg")) or \
				not os.path.exists("%s/%s" % (label_path, file_+"-gt.png")):
				continue

			source = cv2.imread("%s/%s" % (images_path, file_+"-org.jpg"))
			label = cv2.imread("%s/%s" % (label_path, file_+"-gt.png"), 0)

			# height, width = source.shape[:-1]
			# flag = (height - self.__height) + (width - self.__width)
			# interpolation = cv2.INTER_AREA if flag < 0 else cv2.INTER_CUBIC
 			# 
			# source = cv2.resize(source, (self.__width, self.__height), interpolation=interpolation)
			# label = cv2.resize(label, (self.__width, self.__height), interpolation=interpolation)

			# new_label = []
			# 
			# for i in range(0, self.__num_classes):
			# 	array = np.zeros(label.shape)
			# 	array[label == i] = 1
			# 	new_label.append(array)
			# 
			# new_label = np.array(new_label)
			data.append({"source": source, "label": label}) # original

		return data
	
	def get_batch(self, batch, use_gpu=True):
		if use_gpu:
			batch_size = max(1, torch.cuda.device_count()) * batch
		else:
			batch_size = batch
		
		if len(self._data) < batch_size:
			return None

		source = []
		label = []
		random.shuffle(self._index)
		sample = [self._data[idx] for idx in self._index[:batch_size]]

		max_width = 0
		max_height = 0

		for item in sample:
			h, w = item["source"].shape[:-1]
			max_width = max(w, max_width)
			max_height = max(h, max_height)

		for item in sample:
			source_img = item["source"]
			label_img = item["label"]			

			source_img = cv2.resize(source_img, (max_width, max_height), interpolation=cv2.INTER_CUBIC)
			label_img = cv2.resize(label_img, (max_width, max_height), interpolation=cv2.INTER_CUBIC)
			source_img = source_img.transpose((2, 0, 1))

			source.append(source_img)
			label.append(label_img)
		
		source = torch.FloatTensor(source)
		label = torch.LongTensor(label)
	
		return {"source": source, "label": label}


class TrainDataset(DatasetBase):
	
	def __init__(self, path):
		super().__init__(path)
		
		self._data = super()._get_data("training")
		self._index = list(range(len(self._data)))


class ValDataset(DatasetBase):
	
	def __init__(self, path):
		super().__init__(path)
		
		self._data = super()._get_data("validation")
		self._index = list(range(len(self._data)))


class TestDataset:
	
	def __init__(self, path):
		self.__path = path
		self.__data = []

		self.__init_data()

	def __init_data(self):
		for file_ in os.listdir(self.__path):
			try:
				tensor = torch.FloatTensor([cv2.imread("%s/%s" % (self.__path, file_)).transpose((2, 0, 1))])
				self.__data.append({"source": tensor})		
			except:
				pass

	def data(self):
		return self.__data	

	
if __name__ == "__main__":
	dataset = TrainDataset("./dataset")
	batch = dataset.get_batch(1)
