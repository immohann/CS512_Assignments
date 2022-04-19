import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    # def __init__(self, lstm):	# feel free to add more input arguments as needed
        # super(ProximalLSTMCell, self).__init__()
        # self.lstm = lstm   # use LSTMCell as blackbox

    @staticmethod
    def forward(ctx, h_t, s_t, G_t, prox_epsilon=1):
        '''need to be implemented'''
        s_t = s_t.unsqueeze(2)
        G_t_transpose = torch.transpose(G_t, 1, 2)            
        #G_t.permute(1,0)
        mul = torch.matmul(G_t, G_t_transpose)
        my_eye = torch.eye(mul.shape[-1])
        my_eye = my_eye.reshape((1, my_eye.shape[0], my_eye.shape[0]))
        my_eye = my_eye.repeat(h_t.shape[0], 1, 1)
        inverse = torch.inverse(my_eye + prox_epsilon*mul)
        c_t = torch.matmul(inverse, s_t)
        c_t = c_t.squeeze()
        ctx.save_for_backward(h_t, c_t, G_t, inverse)
        return (h_t, c_t)


    @staticmethod
    def backward(ctx, grad_h, grad_c):
        '''need to be implemented'''
        # grad_input = grad_pre_c = grad_pre_h = None
        h_t, c_t, G_t, inverse = ctx.saved_tensors
        # print("inverse.shape: {}, grad_c.shape: {}".format(inverse.shape, grad_c.unsqueeze(2).shape))
        a = torch.matmul(inverse,grad_c.unsqueeze(2))
        grad_1 = torch.matmul(a, c_t.unsqueeze(2).permute(0, 2, 1))
        grad_2 = torch.matmul(c_t.unsqueeze(2), a.permute(0, 2, 1))
        # print("G_t.shape: ", G_t.shape)
        grad_g = -torch.matmul(grad_1 + grad_2, G_t)
        grad_s = torch.matmul(grad_c.unsqueeze(2).permute(0, 2, 1), inverse)
        grad_s = grad_s.squeeze()

        return grad_h, grad_s, grad_g, None