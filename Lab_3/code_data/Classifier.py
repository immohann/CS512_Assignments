
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as ag
from torch.nn import functional as F
import ProxLSTM as pro



class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, input_size):
		super(LSTMClassifier, self).__init__()
		

		self.output_size = output_size	# should be 9
		self.hidden_size = hidden_size  #the dimension of the LSTM output layer
		self.input_size = input_size	  # should be 12
		#self.normalize = F.normalize()
		self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= 64, kernel_size= 5, stride= 2) # feel free to change out_channels, kernel_size, stride
		self.relu = nn.ReLU()
		self.lstm = nn.LSTMCell(64, hidden_size)
		self.lstmcell = nn.LSTMCell(input_size= 64, hidden_size= hidden_size)
		# self.ProxLSTMCell = pro.ProximalLSTMCell(self.lstmcell, input_size= 64, hidden_size= hidden_size)
		# self.ProxLSTMCell = pro.ProximalLSTMCell(self.lstmcell)
		self.linear = nn.Linear(self.hidden_size, self.output_size)
		self.apply_dropout = False
		self.apply_batch_norm = False
		self.dropout = nn.Dropout()
		self.batch_norm = nn.BatchNorm1d(64)

		self.ht = torch.zeros(self.hidden_size, requires_grad= True)		
		self.ct = torch.zeros(self.hidden_size, requires_grad= True)		


		
	def forward(self, input, r, batch_size, mode='plain', prox_epsilon=1, epsilon=0.01):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size


		
		if mode == 'plain':
			norm_inp = F.normalize(input)
			embed_inp = self.conv(norm_inp.permute(0,2,1)).permute(2,0,1)
			self.inp_lstm = self.relu(embed_inp)
			self.ht = torch.zeros(self.inp_lstm.shape[1], self.hidden_size)		
			self.ct = torch.zeros(self.inp_lstm.shape[1], self.hidden_size)		
			for seq in self.inp_lstm:
				self.ht, self.ct = self.lstmcell(seq, (self.ht, self.ct))
			result = self.linear(self.ht)
				# chain up the layers

		if mode == 'AdvLSTM':
			norm_inp = F.normalize(input)
			embed_inp = self.conv(norm_inp.permute(0,2,1)).permute(2,0,1)
			self.inp_lstm = self.relu(embed_inp) + (epsilon * r)
			self.ht = torch.zeros(self.inp_lstm.shape[1], self.hidden_size)		
			self.ct = torch.zeros(self.inp_lstm.shape[1], self.hidden_size)		
			for seq in self.inp_lstm :
				self.ht, self.ct = self.lstmcell(seq, (self.ht, self.ct))
			result = self.linear(self.ht)


		if mode == 'ProxLSTM':
			prox = pro.ProximalLSTMCell.apply
			norm_inp = F.normalize(input)
			# Dropout layer
			if self.apply_dropout:
				norm_inp = self.dropout(norm_inp)
			embed_inp = self.conv(norm_inp.permute(0,2,1)).permute(2,0,1)
			self.inp_lstm = self.relu(embed_inp)
			# Batch Norm layer
			if self.apply_batch_norm:
				self.inp_lstm = self.batch_norm(self.inp_lstm.permute(0, 2, 1))
				self.inp_lstm = self.inp_lstm.permute(0, 2, 1)
			self.ht = torch.zeros(self.inp_lstm.shape[1], self.hidden_size)		
			self.ct = torch.zeros(self.inp_lstm.shape[1], self.hidden_size)		
			for seq in self.inp_lstm:
				self.ht, self.st = self.lstmcell(seq,(self.ht, self.ct))
				self.Gt = torch.zeros(seq.shape[0], self.lstmcell.hidden_size, self.lstmcell.input_size)
				for i in range(self.st.size(-1)):
					gt = ag.grad(self.st[:,i], seq, grad_outputs=torch.ones_like(self.st[:,0]), retain_graph=True)[0]
					self.Gt[:,i,:] = gt[0]
					
				self.ht, self.ct = prox(self.ht, self.st, self.Gt, prox_epsilon)
			result = self.linear(self.ht)
		
		return result
				# chain up layers, but use ProximalLSTMCell here



	def get_inp_lstm(self):
		return self.inp_lstm
