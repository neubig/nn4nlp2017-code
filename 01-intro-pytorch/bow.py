from collections import defaultdict
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

data_dir = "../data/classes/"
train_file = "train.txt"
test_file = "test.txt"
use_cuda = torch.cuda.is_available()

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i['<unk>']

def read_dataset(filename):
	with open(filename, 'r') as f:
		for line in f:
			tag, words = line.lower().strip().split(" ||| ")
			yield ([w2i[x] for x in words.split(" ")], t2i[tag])

train = list(read_dataset(data_dir + train_file))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset(data_dir + test_file))
nwords = len(w2i)
ntags = len(t2i)
print('vocab_size: ' + str(nwords))
print('classes: ' + str(ntags))

class BoW(nn.Module):
	
	def __init__(self, nwords, ntags):
		super(BoW, self).__init__()
		self.embeddings = nn.Embedding(nwords, ntags)
		self.bias = nn.Parameter(torch.ones(ntags))
	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		score = torch.sum(embeds, dim=0, keepdim=True)
		out = score + self.bias
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

losses = []
loss_function = nn.NLLLoss()
model = BoW(nwords, ntags)
optimizer = optim.Adam(model.parameters())

if use_cuda:
	model.cuda()

for epoch in range(100):
	random.shuffle(train)
	total_loss = torch.Tensor([0])
	start = time.time()
	for words, tag in train:
		if use_cuda:
			words_tensor = torch.cuda.LongTensor(words)
			tag_tensor = torch.cuda.LongTensor([tag])
		else:
			words_tensor = torch.LongTensor(words)
			tag_tensor = torch.LongTensor([tag])
		words_var = autograd.Variable(words_tensor)
		tag_var = autograd.Variable(tag_tensor)

		model.zero_grad()

		log_probs = model(words_var)
		
		loss = loss_function(log_probs, tag_var)

		loss.backward()
		optimizer.step()

		total_loss += loss.data

	print("epoch %r: train loss/sent=%.4f, time=%.2fs" % 
		(epoch, total_loss/len(train), time.time()-start))
	
	test_correct = 0.0
	for words, tag in dev:
		if use_cuda:
			words_tensor = torch.cuda.LongTensor(words)
			tag_tensor = torch.cuda.LongTensor([tag])
		else:
			words_tensor = torch.LongTensor(words)
			tag_tensor = torch.LongTensor([tag])
		words_var = autograd.Variable(words_tensor)
		tag_var = autograd.Variable(tag_tensor)

		scores = model(words_var).data.cpu().numpy()
		predict = np.argmax(scores[0])
		if predict == tag:
			test_correct += 1
	print("iter %r: test acc=%.4f" % (epoch, test_correct/len(dev)))