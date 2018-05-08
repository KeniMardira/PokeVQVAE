import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import json

def import_json(path):
    with open(path, 'r') as file:
        hyper_parameters = json.load(file)
    return hyper_parameters

hyper_parameters = import_json('./hyper_parameters.json')

LATENT_DIM = hyper_parameters['LATENT_DIM']
RESIZE = hyper_parameters['RESIZE']

class Net(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Net, self).__init__()
        
        #encoder
        self.conv_e1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn_e1 = nn.BatchNorm2d(32)
        self.conv_e2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn_e2 = nn.BatchNorm2d(64)
        self.conv_e3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn_e3 = nn.BatchNorm2d(128)
        self.conv_e4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn_e4 = nn.BatchNorm2d(256)
        
        self.conv_e5 = nn.Conv2d(256, 64, kernel_size = 1, stride = 1)
        self.bn_e5 = nn.BatchNorm2d(64)
        
        #decoder
        self.up = nn.Upsample(scale_factor=2)
        
        self.conv_d0 = nn.Conv2d(64, 256, kernel_size = 1, stride = 1)
        self.bn_d0 = nn.BatchNorm2d(256)
        
        self.conv_d1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn_d1 = nn.BatchNorm2d(128)
        self.conv_d2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.conv_d3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn_d3 = nn.BatchNorm2d(32)
        self.conv_d4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        self.K = LATENT_DIM
        self.D = 64*RESIZE//16*RESIZE//16
        self.dict = nn.Embedding(self.K, self.D) # K = LATENT_DIM, D = 4096
        
        self.init_weights()        
        
    def init_weights(self):
        initrange = 1.0 / self.K
        self.dict.weight.data.uniform_(-initrange, initrange) 
        
    def encoder(self, x):
        ''' encoder: q(z|x)
            input: x, output: mean, logvar
        '''
        x = F.leaky_relu(self.bn_e1(self.conv_e1(x)))
        x = F.leaky_relu(self.bn_e2(self.conv_e2(x)))
        x = F.leaky_relu(self.bn_e3(self.conv_e3(x)))
        x = F.leaky_relu(self.bn_e4(self.conv_e4(x)))
        x = F.leaky_relu(self.bn_e5(self.conv_e5(x)))
        
        # output b x 64 x 8 x 8
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # z_mean = self.fc_mean(x)
        # z_logvar = self.fc_logvar(x)
        return x
    
    def decoder(self, z):
        '''
            decoder: p(x|z)
            input: z. output: x
        '''
        x = z.view(-1, 64, RESIZE//16, RESIZE//16)
        x = F.leaky_relu(self.bn_d0(self.conv_d0(x)))
        x = F.leaky_relu(self.bn_d1(self.conv_d1(self.up(x))))
        x = F.leaky_relu(self.bn_d2(self.conv_d2(self.up(x))))
        x = F.leaky_relu(self.bn_d3(self.conv_d3(self.up(x))))
        x = F.sigmoid(self.conv_d4(self.up(x)))
        return x.view(-1,3,RESIZE,RESIZE)

    def forward(self, x):
        Z_e = self.encoder(x)
        
        org_Z_e = Z_e
        
        sz = Z_e.size()
        Z_e = Z_e.permute(0,2,3,1)
        Z_e = Z_e.contiguous()
        Z = Z_e.view(-1, self.D) # b x D
        W = self.dict.weight
        
        def L2(a,b):
            return ((a-b)**2)
        
        # sample nearest embedding
        j = L2(Z[:,None], W[None,:]).sum(2).min(1)[1]
        W_j = W[j] # b x D
        
        Z_sg = Z.detach()
        W_j_sg = W_j.detach()
        
        Z_e = W_j.view(sz[0], sz[2], sz[3], sz[1])
        Z_e = Z_e.permute(0,3,1,2)
        
        def hook(grad):
            nonlocal org_Z_e
            self.saved_grad = grad
            self.saved_Z_e = org_Z_e
            return grad
        
        Z_e.register_hook(hook)
        
        return self.decoder(Z_e), L2(Z,W_j_sg).sum(1).mean(), L2(Z_sg,W_j).sum(1).mean()
    
    def bwd(self):
        self.saved_Z_e.backward(self.saved_grad)
        
    