#-*-coding:utf8-*-

import torch

from model import *
from dataset import *
from datetime import datetime


def now():
	return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

n = 2
#model = MyModel2(num_classes=n)
model = torch.load("checkpoint/epoch_28.pth").module

if torch.cuda.device_count() > 0:
	model = nn.DataParallel(model)

if torch.cuda.is_available():
	model = model.cuda()
	
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
print("%s | start to load all dataset" % now())

dataset = TrainDataset("./dataset")

print("%s | end" % now())
print("%s | begin to train" % now())

total_epoch = 300
iters = 100

print("%s | total epoch: %d" % (now(), total_epoch))
print("%s | iters per epoch: %d" % (now(), iters))

for epoch in range(1, total_epoch+1):
	for iter_ in range(1, iters+1):
		batch = dataset.get_batch(6)
		loss = model(batch)
		loss = loss.mean()
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		lr = optimizer.param_groups[0]['lr']
		print("%s | epoch: %d, iter: %d, loss: %.3f, lr: %.3f" % (now(), epoch, iter_, loss, lr))
	
	torch.save(model, "checkpoint/epoch_%d.pth" % epoch)

torch.save(model, "checkpoint/last.pth")
print("%s | end trainning" % now())
