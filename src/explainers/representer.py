#!/usr/bin/env python
# coding: utf-8
import os.path
from time import time
import sys
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch
from explainers import FeatureKernelExplainer

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class softmax(nn.Module):
    def __init__(self, W, device):
        if device=="cuda":
            data_type = torch.cuda.FloatTensor
        else:
            data_type = torch.FloatTensor
        super(softmax, self).__init__()
        self.W = Variable(W.type(data_type), requires_grad=True)

    def forward(self, x, y):
        # calculate loss for the loss function and L2 regularizer
        D = (torch.matmul(x,self.W))
        D_max,_ = torch.max(D,dim = 1, keepdim = True)
        D = D-D_max
        A = torch.log(torch.sum(torch.exp(D),dim = 1))
        B = torch.sum(D*y,dim=1)
        Phi = torch.sum(A-B)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return (Phi,L2)

def softmax_np(x):
    e_x = np.exp(x - np.max(x,axis = 1,keepdims = True))
    return e_x / e_x.sum(axis = 1,keepdims = True)

def to_np(x):
    return x.data.cpu().numpy()

# calculation for softmax in torch, which avoids numerical overflow
def softmax_torch(temp,N):
    max_value,_ = torch.max(temp,1,keepdim = True)
    temp = temp-max_value
    D_exp = torch.exp(temp)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N,1)
    return D_exp.div(D_exp_sum.expand_as(D_exp))


class RepresenterPointsExplainer(FeatureKernelExplainer):
    lmbd=0.003
    epoch=3000
    # implementation for backtracking line search
    @staticmethod
    def backtracking_line_search(optimizer, model, grad, x, y, val, beta, N,device):
        if device=="cuda":
            data_type = torch.cuda.FloatTensor
        else:
            data_type = torch.FloatTensor
        t = 10.0
        beta = 0.5
        W_O = to_np(model.W)
        grad_np = to_np(grad)
        cont=True
        while (True):
            model.W = Variable(torch.from_numpy(W_O - t * grad_np).type(data_type), requires_grad=True)
            val_n = 0.0
            (Phi, L2) = model(x, y)
            val_n = Phi / N + L2 * RepresenterPointsExplainer.lmbd
            if t < 0.0000000001:
                break
                cont=False
                print(f"t={t}")
            if to_np(val_n - val + t * torch.norm(grad) ** 2 / 2) >= 0:
                t = beta * t
            else:
                break
                cont=False
                print("val_n - val + t * grad**2 < 0 ")

    name = "RepresenterPointsExplainer"


    def __init__(self, model, dataset, device):
        super(RepresenterPointsExplainer, self).__init__(model, dataset, device, normalize=False)

    def train(self):
        t0=time()
        X=self.samples
        Y=softmax_torch(self.model.classifier(X), X.shape[0])
        model=softmax(self.model.classifier.weight.data.T,self.device)
        X=X.to("cpu")
        Y=Y.to("cpu")
        if self.device=="cpu":
            x = Variable(torch.FloatTensor(X))
            y = Variable(torch.FloatTensor(Y))
        else:
            x = Variable(torch.FloatTensor(X).cuda())
            y = Variable(torch.FloatTensor(Y).cuda())

        N = len(Y)
        min_loss = 10000.0
        optimizer = optim.SGD([model.W], lr=1.0)
        for epoch in range(RepresenterPointsExplainer.epoch):
            sum_loss = 0
            phi_loss = 0
            optimizer.zero_grad()
            (Phi, L2) = model(x, y)
            loss = L2 * RepresenterPointsExplainer.lmbd + Phi / N
            phi_loss += to_np(Phi / N)
            loss.backward()
            temp_W = model.W.data
            grad_loss = to_np(torch.mean(torch.abs(model.W.grad)))
            # save the W with lowest loss
            if grad_loss < min_loss:
                if epoch == 0:
                    init_grad = grad_loss
                min_loss = grad_loss
                best_W = temp_W
                if min_loss < init_grad / 200:
                    print('stopping criteria reached in epoch :{}'.format(epoch))
                    break
            RepresenterPointsExplainer.backtracking_line_search(optimizer, model, model.W.grad, x, y, loss, 0.5, N, self.device)
            if epoch % 100 == 0:
                print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(epoch, to_np(loss), phi_loss, grad_loss))

        elapsed_time=time()-t0
        # calculate w based on the representer theorem's decomposition
        temp = torch.matmul(x, Variable(best_W))
        self.learned_weight=best_W.T
        softmax_value = softmax_torch(temp, N)
        # derivative of softmax cross entropy
        weight_matrix = softmax_value - y
        weight_matrix = torch.div(weight_matrix, (-2.0 * RepresenterPointsExplainer.lmbd * N))
        print(weight_matrix[:5, :5].cpu())
        w = torch.matmul(torch.t(x), weight_matrix)
        print(w[:5, :5].cpu())
        # calculate y_p, which is the prediction based on decomposition of w by representer theorem
        if self.device=="cpu":
            temp = torch.matmul(x, w)
        else:
            temp = torch.matmul(x, w.cuda())
        print(temp[:5, :5].cpu())
        softmax_value = softmax_torch(temp, N)
        y_p = to_np(softmax_value)
        print(y_p[:5, :])

        print('L1 difference between ground truth prediction and prediction by representer theorem decomposition')
        print(np.mean(np.abs(to_np(y) - y_p)))

        from scipy.stats.stats import pearsonr
        print('pearson correlation between ground truth  prediction and prediciton by representer theorem')
        y = to_np(y)
        corr, _ = (pearsonr(y.flatten(), (y_p).flatten()))
        print(corr)
        sys.stdout.flush()
        self.coefficients = weight_matrix
        #return weight_matrix
        return elapsed_time
