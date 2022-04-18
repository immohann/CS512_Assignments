
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

		self.h_t = torch.zeros(self.hidden_size, requires_grad= True)		# h_0
		self.c_t = torch.zeros(self.hidden_size, requires_grad= True)		# c_0


		
	def forward(self, input, r, batch_size, mode='plain', prox_epsilon=1, epsilon=0.01):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size


		'''need to be implemented'''
		if mode == 'plain':
			norm_inp = F.normalize(input)
			embed_inp = self.conv(norm_inp.permute(0,2,1)).permute(2,0,1)
			self.lstm_input = self.relu(embed_inp)
			self.h_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# h_0
			self.c_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# c_0
			for seq in self.lstm_input:
				self.h_t, self.c_t = self.lstmcell(seq, (self.h_t, self.c_t))
			result = self.linear(self.h_t)
				# chain up the layers

		if mode == 'AdvLSTM':
			norm_inp = F.normalize(input)
			embed_inp = self.conv(norm_inp.permute(0,2,1)).permute(2,0,1)
			self.lstm_input = self.relu(embed_inp) + (epsilon * r)
			self.h_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# h_0
			self.c_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# c_0
			for seq in self.lstm_input :
				self.h_t, self.c_t = self.lstmcell(seq, (self.h_t, self.c_t))
			result = self.linear(self.h_t)
				# chain up the layers
			  # different from mode='plain', you need to add r to the forward pass
			  # also make sure that the chain allows computing the gradient with respect to the input of LSTM

		if mode == 'ProxLSTM':
			prox = pro.ProximalLSTMCell.apply
			norm_inp = F.normalize(input)
			# Dropout layer
			if self.apply_dropout:
				norm_inp = self.dropout(norm_inp)
			embed_inp = self.conv(norm_inp.permute(0,2,1)).permute(2,0,1)
			self.lstm_input = self.relu(embed_inp)
			# Batch Norm layer
			if self.apply_batch_norm:
				self.lstm_input = self.batch_norm(self.lstm_input.permute(0, 2, 1))
				self.lstm_input = self.lstm_input.permute(0, 2, 1)
			self.h_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# h_0
			self.c_t = torch.zeros(self.lstm_input.shape[1], self.hidden_size)		# c_0
			for seq in self.lstm_input:
				self.h_t, self.s_t = self.lstmcell(seq,(self.h_t, self.c_t))
				self.G_t = torch.zeros(seq.shape[0], self.lstmcell.hidden_size, self.lstmcell.input_size)
				for i in range(self.s_t.size(-1)):
					g_t = ag.grad(self.s_t[:,i], seq, grad_outputs=torch.ones_like(self.s_t[:,0]), retain_graph=True)[0]
					# print("g_t: ", g_t)
					self.G_t[:,i,:] = g_t[0]
					
				# print("G_t.shape", self.G_t.shape)
				self.h_t, self.c_t = prox(self.h_t, self.s_t, self.G_t, prox_epsilon)
			result = self.linear(self.h_t)
		
		return result
				# chain up layers, but use ProximalLSTMCell here



	def get_lstm_input(self):
		return self.lstm_input
