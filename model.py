#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, xavier_uniform_

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
	#lossfn = nn.BCEWithLogitsLoss().cuda()
	#lossfn = nn.HingeEmbeddingLoss(reduction = 'sum')
	lossfn = nn.SoftMarginLoss()
	def __init__(self, vocab_size, embedding_size, gamma, batch_size, neg_ratio, dpo1, dpo2,bn1,bn2, channel1_num=10,
				 channel2_num=10):
		super(KGEModel, self).__init__()
		self.entity_embed = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0).cuda()
		self.fc2size = embedding_size * channel1_num
		self.embedding_size = embedding_size
		self.dpo1 = dpo1
		self.dpo2 = dpo2
		self.bn1 = bn1
		self.bn2 = bn2
		self.batch_size = batch_size
		'''self.gamma = nn.Parameter(
			torch.Tensor([gamma]),
			requires_grad=False
		)'''

		# self.lossfn = nn.BCELoss().cuda()
		self.channel1_num = channel1_num
		self.channel2_num = channel2_num
		self.dropout = nn.Dropout2d(p=0.4)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=channel1_num, kernel_size=(1, 2)).cuda()
		# self.softplus = nn.Softplus()
		self.bnorm1 = nn.BatchNorm2d(num_features=channel1_num)
		self.fc2 = nn.Linear(in_features=self.fc2size, out_features=1, bias=True).cuda()

	def init(self):
		xavier_normal_(self.emb_e.weight.data)
		xavier_normal_(self.emb_rel.weight.data)

	def calVec(self, conv_net, v, valid_check):
		v = conv_net(v)
		if self.dpo1 is True and valid_check is False:
			v = self.dropout(v)
		# print('v val <= -3: ',v[v<=-3].shape)
		# print('after convolution')
		# print(v[v!=0].shape)
		# print(v[v == 0].shape)
		if self.bn1 is True and valid_check is False:
			v = self.bnorm1(v)
		v = F.relu(v)

		return v #final_v.float().cuda()  # .view(-1, 1).float().cuda()

	def forward(self, h, r, t,valid_check):
		h = self.entity_embed(h).view(-1, 1, self.embedding_size, 1)
		t = self.entity_embed(t).view(-1, 1, self.embedding_size, 1)  # .view(-1,1)
		r = self.entity_embed(r).view(-1, 1, self.embedding_size, 1)

		# print(h)
		# print('r tensor:', r)
		# print('t tensor:', t)
		# exit(0)

		alpha = 0.5
		rh = alpha*r
		rt = (1-alpha)*r

		hrh = torch.cat((h,rh),dim=3)
		trt = torch.cat((rt,t),dim=3)

		# v1 = self.calVec(self.conv1, hrh, valid_check).cuda().view(-1,1,self.embedding_size*self.channel1_num,1)
		# v2 = self.calVec(self.conv2, trt, valid_check).cuda().view(-1,1,self.embedding_size*self.channel1_num,1)
		v1 = self.calVec(self.conv1, hrh, valid_check).cuda().view(-1,self.embedding_size * self.channel1_num)
		v2 = self.calVec(self.conv1, trt, valid_check).cuda().view(-1,self.embedding_size*self.channel1_num)

		finalTensor = torch.abs(v1-v2)
		#print(finalTensor.shape)
		finalTensor = self.fc2(finalTensor.reshape(-1, self.embedding_size*self.channel1_num))
		return finalTensor

	def ConvKBloss(self,h,r,t,y_val,valid_check,optimizer,model):
		loss = model(h, r, t,valid_check) * torch.tensor(y_val, requires_grad=True).cuda()
		# l2_reg = Variable(torch.FloatTensor(1), requires_grad=True).cuda()

		# for W in model.parameters():
		#	l2_reg = l2_reg + W.norm(2)

		loss = F.softplus(loss)
		#W = self.fc.weight
		#b = self.fc.bias
		# print(W)
		# print(W.shape)
		# exit(0)
		loss = torch.mean(loss) #+ 0.0001/2*(torch.sum(W**2)+torch.sum(b**2)) #+ 0.0001*l2_reg/2 #0.0001 is lambda
		loss = loss.cuda()
		if valid_check is False:  # do training, not validation
			loss.backward()
			optimizer.step()
		return loss.item(),optimizer

	#@staticmethod
	def train_step(self,model, optimizer, train_batch, valid_check=False):
		'''
		A single train step. Apply back-propation and return the loss
		'''

		optimizer.zero_grad()

		triple, y_val, subsampling_weight = train_batch()

		half_sample_num = int(triple.shape[0] / 2)
		# print(half_sample_num)
		# exit(0)
		# print('y_val shape:', y_val.shape)
		# print('triple',triple.shape)

		# may divide this to positive and negative samples -> feed to forward function
		h = torch.tensor(triple[:, 0], dtype=torch.long).cuda()
		r = torch.tensor(triple[:, 1], dtype=torch.long).cuda()
		t = torch.tensor(triple[:, 2], dtype=torch.long).cuda()
		y_val = torch.tensor(y_val).cuda()  # .view(-1,1).cuda() #may not need


		#if valid_check is False:
		#	y_val = torch.cat((y_val,y_val),dim=0) # to generate inversion relation
		#subsampling_weight = torch.tensor(subsampling_weight).reshape(-1, 1).cuda()


		'''
		ConvE loss (not work T-T)
		'''
		#loss, optimizerret = self.ConvKBloss(h, r, t, y_val, valid_check, optimizer, model)
		#return loss, optimizerret
		'''
		ConvKB loss function
		'''

		loss, optimizerret = self.ConvKBloss(h,r,t,y_val,valid_check,optimizer,model)
		#print(loss)
		return loss, optimizerret

	@staticmethod
	def test_step(model, test_triples, all_true_triples, args):
		'''
		Evaluate the model on test or valid datasets
		'''

		model.eval()

		if args.countries:
			# Countries S* datasets are evaluated on AUC-PR
			# Process test data for AUC-PR evaluation
			sample = list()
			y_true = list()
			for head, relation, tail in test_triples:
				for candidate_region in args.regions:
					y_true.append(1 if candidate_region == tail else 0)
					sample.append((head, relation, candidate_region))

			sample = torch.LongTensor(sample)
			if args.cuda:
				sample = sample.cuda()

			with torch.no_grad():
				y_score = model(sample).squeeze(1).cpu().numpy()

			y_true = np.array(y_true)

			# average_precision_score is the same as auc_pr
			auc_pr = average_precision_score(y_true, y_score)

			metrics = {'auc_pr': auc_pr}

		else:
			# Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
			# Prepare dataloader for evaluation
			test_dataloader_head = DataLoader(
				TestDataset(
					test_triples,
					all_true_triples,
					args.nentity,
					args.nrelation,
					'head-batch'
				),
				batch_size=args.test_batch_size,
				num_workers=max(1, args.cpu_num // 2),
				collate_fn=TestDataset.collate_fn
			)

			test_dataloader_tail = DataLoader(
				TestDataset(
					test_triples,
					all_true_triples,
					args.nentity,
					args.nrelation,
					'tail-batch'
				),
				batch_size=args.test_batch_size,
				num_workers=max(1, args.cpu_num // 2),
				collate_fn=TestDataset.collate_fn
			)

			test_dataset_list = [test_dataloader_head, test_dataloader_tail]

			logs = []

			step = 0
			total_steps = sum([len(dataset) for dataset in test_dataset_list])

			with torch.no_grad():
				for test_dataset in test_dataset_list:
					for positive_sample, negative_sample, filter_bias, mode in test_dataset:
						if args.cuda:
							positive_sample = positive_sample.cuda()
							negative_sample = negative_sample.cuda()
							filter_bias = filter_bias.cuda()

						batch_size = positive_sample.size(0)

						score = model((positive_sample, negative_sample), mode)
						score += filter_bias

						# Explicitly sort all the entities to ensure that there is no test exposure bias
						argsort = torch.argsort(score, dim=1, descending=True)

						if mode == 'head-batch':
							positive_arg = positive_sample[:, 0]
						elif mode == 'tail-batch':
							positive_arg = positive_sample[:, 2]
						else:
							raise ValueError('mode %s not supported' % mode)

						for i in range(batch_size):
							# Notice that argsort is not ranking
							ranking = (argsort[i, :] == positive_arg[i]).nonzero()
							assert ranking.size(0) == 1

							# ranking + 1 is the true ranking used in evaluation metrics
							ranking = 1 + ranking.item()
							logs.append({
								'MRR': 1.0 / ranking,
								'MR': float(ranking),
								'HITS@1': 1.0 if ranking <= 1 else 0.0,
								'HITS@3': 1.0 if ranking <= 3 else 0.0,
								'HITS@10': 1.0 if ranking <= 10 else 0.0,
							})

						if step % args.test_log_steps == 0:
							logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

						step += 1

			metrics = {}
			for metric in logs[0].keys():
				metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

		return metrics