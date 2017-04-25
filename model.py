import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from huva.th_util import *


def gumbel_noise(x, eps=1e-10):
    """ 
    x is [N, ]
    """
    noise = x.new().resize_as_(x).uniform_(0, 1)
    return -(-(noise + eps).log() + eps).log()

def gumbel_max(x):
    noisy_x = x + gumbel_noise(x)
    max_x = noisy_x.max(1)[0].expand_as(x)
    mask_x = max_x == noisy_x
    return mask_x

def plain_max(x):
    max_x = x.max(1)[0].expand_as(x)
    mask_x = max_x == x
    return mask_x

class CatFactor(nn.Module):
    def __init__(self, num_factor, num_category, 
            nonclass=False, fix_bias=False, stochastic=False):
        """
        num_factor: number of factors
        num_category: number of categories per factor
        nonclass:
            normally the output of a softmax sum to 1. When we include a non-class element, the output will sum to less
            than 1. In this case, the outputs are considered as "variants" of the factor, whereas the non-class element
            is considered "factor not present". This non-class element has fixed log probability, which is a trainable
            parameter independent of input.
        """
        nn.Module.__init__(self)
        self.num_factor   = num_factor
        self.num_category = num_category
        self.nonclass     = nonclass
        self.fix_bias     = fix_bias
        self.stochastic   = stochastic
        if nonclass:
            logprob = torch.Tensor(1, num_factor, 1)
            # logprob.normal_(0, 1)
            logprob.fill_(5)
            if fix_bias:
                self.register_buffer('nonclass_logprob', Variable(logprob))
            else:
                self.nonclass_logprob = nn.Parameter(logprob)

    def forward(self, x):
        """
        x        : N, num_factor * num_category
        reshaped : N * num_factor, num_category
        softmaxed: N * num_factor, num_category
        """
        assert x.dim()==2
        assert x.size(1)== self.num_factor * self.num_category
        N = x.size(0)

        if self.nonclass:
            """
            1. reshape as [N, num_factor, num_category]
            2. append nonclass_logprob to above, giving [N, num_factor, num_category+1]
            3. softmax on last dimension, giving [N*num_factor, num_category+1]
            4. drop last element of last dimension, giving [N*num_factor, num_category]
            """
            reshaped  = x.view(N, self.num_factor, self.num_category)
            expanded_logprob = self.nonclass_logprob.expand(N, self.num_factor, 1)
            reshaped  = torch.cat([reshaped, expanded_logprob], 2)
            reshaped  = reshaped.view(N * self.num_factor, self.num_category + 1)
            if self.stochastic:
                result = reshaped * Variable(gumbel_max(reshaped.data).float())
            else:
                result = F.softmax(reshaped)
            # if self.output_sum:
                # assert not self.stochastic
                # softmaxed = F.softmax(reshaped)
                # softmaxed[:, -1] = 1 - softmaxed[:, -1]
            # else:
            result = result[:, :-1].contiguous()
        else:
            reshaped  = x.view(N * self.num_factor, self.num_category)
            if self.stochastic:
                result = reshaped * Variable(gumbel_max(reshaped.data).float())
            else:
                result = F.softmax(reshaped)

        return result.view_as(x)

    def __repr__(self):
        return "CatFactor (factors={}, categories={}, nonclass={}, stochastic={})".format(
                self.num_factor, self.num_category, self.nonclass, self.stochastic)

class CatFactor2d(nn.Module):
    def __init__(self, num_factor, num_category, stochastic=False):
        nn.Module.__init__(self)
        self.num_factor   = num_factor
        self.num_category = num_category
        self.stochastic   = stochastic
    def forward(self, x):
        assert x.dim()==4
        assert x.size(1)== self.num_factor * self.num_category
        N = x.size(0)
        reshaped = x.view(N * self.num_factor, self.num_category, x.size(2), x.size(3))
        if self.stochastic:
            result = reshaped * Variable(plain_max(reshaped.data).float())
        else:
            result = F.softmax(reshaped) # F.softmax works for 2D and 4D tensors, taking softmax across second dim

        return result.view_as(x)
    def __repr__(self):
        return "CatFactor (factors={}, categories={}, stochastic={})".format(
                self.num_factor, self.num_category, self.stochastic)


class MatrixD(nn.Module):
    def __init__(self, num_factor, num_category, std=1, factor_mode=False):
        nn.Module.__init__(self)
        self.num_factor   = num_factor
        self.num_category = num_category
        self.std = std
        self.factor_mode = factor_mode
        self.set_D(std)
    def set_D(self, std):
        if self.factor_mode:
            guidance_matrix = torch.zeros(self.num_category, self.num_category)
            for i in range(self.num_category):
                for j in range(self.num_category):
                    guidance_matrix[i,j] = math.exp(-(float(i-j)**2 / std**2))
        else:
            num_unit = self.num_factor * self.num_category
            guidance_matrix = torch.zeros(num_unit, num_unit)
            for group in range(self.num_factor):
                for i in range(self.num_category):
                    ii = group * self.num_category + i
                    for j in range(self.num_category):
                        jj = group * self.num_category + j
                        guidance_matrix[ii,jj] = math.exp(-(float(i-j)**2 / std**2))
        self.std = std
        self.register_buffer('D', Variable(guidance_matrix))
    def forward(self, x):
        num_unit = self.num_factor * self.num_category
        assert x.size(1) == num_unit
        if self.factor_mode:
            result = x.clone()
            for i in xrange(0, num_unit, self.num_category):
                k = x[:, i:i+self.num_category].mm(self.D)
                result[:, i:i+self.num_category] = k
            return result
        else:
            return x.mm(self.D)
    def __repr__(self):
        return "MatrixD (factors={}, categories={}, std={})".format(
                self.num_factor, self.num_category, self.std)

class MatrixD2d(nn.Module):
    def __init__(self, num_factor, num_category, std=1, decay=0.995, decay_interval=10):
        nn.Module.__init__(self)
        self.num_factor   = num_factor
        self.num_category = num_category
        self.std = std
        self.num_batches = 0
        self.decay = decay
        self.decay_interval = decay_interval
        self.set_D(std)
    def set_D(self, std):
        num_unit = self.num_factor * self.num_category
        nc = self.num_category
        """ initialize buffer """
        if hasattr(self, 'D'):
            guidance_matrix = self.D.data.fill_(0)
        else:
            guidance_matrix = torch.zeros(num_unit, num_unit, 1, 1)
        """ set up D """
        for i in range(nc):
            for j in range(nc):
                guidance_matrix[i,j, 0, 0] = math.exp(-(float(i-j)**2 / std**2))
        """ normalize each column """
        summed = guidance_matrix[:nc, :nc].sum(0)
        guidance_matrix[:nc, :nc].div_(summed.expand(nc, nc, 1, 1))
        """ duplicate for all groups """
        for group in range(1, self.num_factor):
            #print(guidance_matrix[group*nc: (group+1)*nc].size(), guidance_matrix[:nc, :nc].size())
            guidance_matrix[group*nc: (group+1)*nc, group*nc:(group+1)*nc] = guidance_matrix[:nc, :nc]
        """ register things """
        self.std = std
        self.register_buffer('D', Variable(guidance_matrix))

    def forward(self, x):
        self.num_batches += 1
        if self.num_batches % self.decay_interval == 0:
            self.std *= self.decay
            self.set_D(self.std)
        return F.conv2d(x, self.D)

    def __repr__(self):
        return "MatrixD (factors={}, categories={}, std={})".format(
                self.num_factor, self.num_category, self.std)

def make_mlpae(num_input, num_factor, num_category):
    """ naive mlpae """
    num_units = num_factor * num_category
    layers = [
        nn.Linear(num_input, num_units),
        CatFactor(num_factor, num_category, nonclass=True, fix_bias=True),
        nn.Linear(num_units, num_input) 
    ]
    return nn.Sequential(*layers)

def make_mlpae_s(num_input, num_factor, num_category):
    """ stochastic naive mlpae """
    num_units = num_factor * num_category
    layers = [
        nn.Linear(num_input, num_units),
        CatFactor(num_factor, num_category, nonclass=True, fix_bias=True, stochastic=True),
        nn.Linear(num_units, num_input) 
    ]
    return nn.Sequential(*layers)

def make_mlpae_d(num_input, num_factor, num_category):
    """ stochastic naive mlpae with matrix D and batchnorm """
    num_units = num_factor * num_category
    layers = [
        nn.Linear(num_input, num_units),
        MatrixD(num_factor, num_category, std=1),
        nn.BatchNorm1d(num_units),
        CatFactor(num_factor, num_category, nonclass=True, fix_bias=True),
        nn.Linear(num_units, num_input) 
    ]
    return nn.Sequential(*layers)

def make_mlpae_ds(num_input, num_factor, num_category):
    """ stochastic naive mlpae with matrix D and batchnorm """
    num_units = num_factor * num_category
    layers = [
        nn.Linear(num_input, num_units),
        MatrixD(num_factor, num_category, std=1),
        nn.BatchNorm1d(num_units),
        CatFactor(num_factor, num_category, nonclass=True, fix_bias=True, stochastic=True),
        nn.Linear(num_units, num_input) 
    ]
    return nn.Sequential(*layers)

def make_mlpae_bn(num_input, num_factor, num_category):
    """ naive mlpae with batchnorm """
    num_units = num_factor * num_category
    layers = [
        nn.Linear(num_input, num_units),
        nn.BatchNorm1d(num_units),
        CatFactor(num_factor, num_category, nonclass=True, fix_bias=True),
        nn.Linear(num_units, num_input) 
    ]
    return nn.Sequential(*layers)

def make_mlpae_bns(num_input, num_factor, num_category):
    """ naive mlpae with batchnorm """
    num_units = num_factor * num_category
    layers = [
        nn.Linear(num_input, num_units),
        nn.BatchNorm1d(num_units),
        CatFactor(num_factor, num_category, nonclass=True, fix_bias=True, stochastic=True),
        nn.Linear(num_units, num_input) 
    ]
    return nn.Sequential(*layers)

def make_mlpae_simple(num_input, num_hidden):
    """ without categorical decomposition """
    layers = [
        nn.Linear(num_input, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_input)
    ]
    return nn.Sequential(*layers)

def get_cnn_params(k, stride):
    pad    = (k-1)/2
    out_padding = (28 + 2*pad - k) % stride
    return k, stride, pad, out_padding

def make_cnnae(num_input, num_factor, num_category):
    """ naive cnnae """
    num_units = num_factor * num_category
    k, stride, pad, out_padding = get_cnn_params(5, 2)
    layers = [
        nn.Conv2d(num_input, num_units, (k,k), stride=stride, padding=pad),
        CatFactor2d(num_factor, num_category),
        nn.ConvTranspose2d(num_units, num_input, (k,k), stride=stride, padding=pad, output_padding=out_padding) 
    ]
    return nn.Sequential(*layers)

def make_cnnae_d(num_input, num_factor, num_category):
    """ naive cnnae """
    num_units = num_factor * num_category
    k, stride, pad, out_padding = get_cnn_params(5, 2)
    layers = [
        nn.Conv2d(num_input, num_units, (k,k), stride=stride, padding=pad),
        MatrixD2d(num_factor, num_category, std=20),
        nn.BatchNorm2d(num_units),
        CatFactor2d(num_factor, num_category),
        nn.ConvTranspose2d(num_units, num_input, (k,k), stride=stride, padding=pad, output_padding=out_padding)
    ]
    return nn.Sequential(*layers)

def make_cnnae_ds(num_input, num_factor, num_category):
    """ naive cnnae """
    num_units = num_factor * num_category
    k, stride, pad, out_padding = get_cnn_params(9, 3)
    layers = [
        nn.Conv2d(num_input, num_units, (k,k), stride=stride, padding=pad),
        MatrixD2d(num_factor, num_category, std=2.5, decay=0.9985),
        #nn.BatchNorm2d(num_units),
        CatFactor2d(num_factor, num_category, stochastic=True),
        nn.ConvTranspose2d(num_units, num_input, (k,k), stride=stride, padding=pad, output_padding=out_padding)
    ]
    return nn.Sequential(*layers)

def make_cnnae_simple(num_input, num_hidden):
    k, stride, pad, out_padding = get_cnn_params(9, 3)
    layers = [
        nn.Conv2d(num_input, num_hidden, (k,k), stride=stride, padding=pad),
        nn.ReLU(),
        nn.ConvTranspose2d(num_hidden, num_input, (k,k), stride=stride, padding=pad, output_padding=out_padding)
    ]
    return nn.Sequential(*layers)

