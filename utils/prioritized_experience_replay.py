# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import random
from sum_tree import SumTree


class PrioritizedExperienceReplay(object):

	def __init__(self, maxlen, input_shape, action_size, eps, alpha):
		self.maxlen = maxlen
		# use a sum tree to store the indices to each transition
		self.e = eps
		self.alpha = alpha
		self.tree = SumTree(maxlen)
		self.max_priority = -99999
		self.min_priority = 99999
		# where we'll store the mem map
		dirname = tempfile.mkdtemp()
		#use memory maps so we won't have to worry about eating up lots of RAM
		get_path = lambda name: os.path.join(dirname, name)
		self.screens = np.memmap(get_path('screens'), dtype=np.float32, mode='w+', shape=tuple([self.maxlen]+input_shape))
		self.actions = np.memmap(get_path('actions'), dtype=np.float32, mode='w+', shape=(self.maxlen, action_size))
		self.rewards = np.memmap(get_path('rewards'), dtype=np.float32, mode='w+', shape=(self.maxlen,))
		self.is_terminal = np.memmap(get_path('terminals'), dtype=np.bool, mode='w+', shape=(self.maxlen,))

		self.position = 0
		self.full = False


	def _get_priority(self,error):
		p = (error + self.e) ** self.alpha
		if p > self.max_priority:
			self.max_priority = p
		if p < self.min_priority:
			self.min_priority = p
		return p

	def sample_batch(self, batch_size):
		batch_idx = np.zeros((batch_size,), dtype=np.int32)
		batch_p = np.zeros((batch_size, ), dtype=np.float32)
		batch_pos = np.zeros((batch_size,),dtype=np.int32)
		idx = 0
		segment = self.tree.total() / batch_size
		P_i = np.zeros((batch_size,), dtype=np.float32)

		for i in range(batch_size):
			a = segment * i
			b = segment * (i + 1)
			s = random.uniform(a,b)
			(idx, p, pos) = self.tree.get(s)
			batch_idx[i] = idx
			batch_p[i] = p
			batch_pos[i] = pos

		# s_i, s_f = self._get_state(batch)
		s_i = self.screens[batch_pos]
		s_f = self.screens[batch_pos+1]
		a = self.actions[batch_pos]
		r = self.rewards[batch_pos]
		is_terminal = self.is_terminal[batch_pos+1]

		for i in range(batch_size):
			P_i[i] = batch_p[i] / self.tree.total()
		P_min = self.min_priority / self.tree.total()
		assert P_min <= 1, "P_max violates cdf"
		'''
		'''
		return s_i, a, r, s_f, is_terminal, batch_idx, P_i, P_min

	def __len__(self):
		return self.maxlen if self.full else self.position

	def append(self, s_i, a, r, is_terminal, error):
		p = self._get_priority(error)
		p = self.max_priority # acording to the paper, we store new transitions with the max priority

		self.screens[self.position] = s_i
		self.actions[self.position] = a
		self.rewards[self.position] = r
		self.is_terminal[self.position] = is_terminal

		if self.position + 1 == self.maxlen:
			self.full = True
		self.position = (self.position + 1) % self.maxlen

		self.tree.add(p,self.position)

	def update(self, idx, error):
		p = self._get_priority(error)
		self.tree.update(idx, p)
