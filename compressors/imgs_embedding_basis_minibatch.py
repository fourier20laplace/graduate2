import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


''' Compressor generator '''
class Net(nn.Module):
    def __init__(
            self,
            ipc,
            channel,
            im_size,
            num_classes,
            device,
            n_basis,
            downsample_scale=1,
            n_per_c=None,
            real_init=None
        ):
        super(Net, self).__init__()
        self.name    = 'imgs_embedding_basis_minibatch'
        self.n_basis = n_basis
        self.ipc     = ipc
        self.n_per_c = n_per_c
        self.channel = channel
        self.im_size = [int(ele / downsample_scale) for ele in im_size]#! basis经过了downsample_scale倍下采样 后面采样出image后还要再恢复
        self.downsample_scale = downsample_scale
        self.num_classes = num_classes
        self.device      = device

        self.prev_imgs   = None

        self.basis = nn.Embedding(self.n_basis, channel*np.prod(self.im_size))#n_basis个basis
        self.imgs  = nn.Embedding(ipc*num_classes, self.n_basis)#ipc个矩阵A 矩阵A大小为num_classes*n_basis
        self.labs  = torch.tensor(
                         np.array([np.ones(self.ipc)*i for i in range(self.num_classes)]),
                         requires_grad=False,
                         device=self.device
                     ).long().view(-1)#蒸馏一开始输入的y 无需独热编码 因为前面是embedding? 
        #以及可以看到这个标签实际上是要获取所有类别的蒸馏数据 大小也为ipc*num_classes

        torch.nn.init.xavier_uniform(self.imgs.weight)
        torch.nn.init.xavier_uniform(self.basis.weight)
        
        print(">"*100)
        print(f"hello it's me {self.name}")
        print(f"Compressor Total parameters: {sum(p.numel() for p in self.parameters())}")
        # 各参数名及参数量
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel()}")
        print(">"*100)

    def get_compressors(self):#? 这个函数应该是没用 作者把interpolate过程显式的写在了外面？
        imgs, _, _ = self.forward(combine=True, new_batch_size=self.ipc)
        imgs = F.interpolate(
                   imgs,
                   scale_factor=self.downsample_scale,
                   mode='bilinear',
                   align_corners=False
               )

        return imgs

    def get_basis(self):
        basis = self.basis(torch.arange(self.n_basis).to(self.device))
        basis = basis.view(
                    self.n_basis,
                    self.channel,
                    self.im_size[0],
                    self.im_size[1]
                ).contiguous()
# 感觉意思·是用nn.embedding初始化 但在这里把他又取出来作为张量形式？
        return basis

    def combine_basis(self, basis, coefficients):
        N, C, H, W = coefficients.shape[0], basis.shape[1], basis.shape[2], basis.shape[3]
        imgs       = torch.matmul(coefficients, basis.view(coefficients.shape[1],-1)).view(N, C, H, W)
        return imgs#？哦没事了眼花了 他这是matmul 【N,n_basis】*【n_basis,channel*H*W】 之后再view的

    def assign_grads(self, grads, task_indices):
        assert isinstance(grads, list)
        basis_grads, imgs_grads = grads
        self.basis.weight.grad  = basis_grads.to(self.basis.weight.data.device).view(self.basis.weight.shape)
        self.imgs.weight.grad   = imgs_grads.to(self.imgs.weight.data.device).view(self.imgs.weight.shape)

    def get_coeffs_min_max(self):
        indices = torch.randperm(self.ipc*self.num_classes).to(self.device)
        coeffs  = self.imgs(indices)
        return coeffs.min().item(), coeffs.max().item()

    def get_coeffs_per_class(self):
        indices = torch.arange(self.ipc*self.num_classes).to(self.device)
        coeffs  = self.imgs(indices)
        coeffs  = coeffs.view(self.num_classes, self.ipc, coeffs.shape[-1])
        return coeffs

    def get_min_max(self):
        indices = torch.randperm(self.ipc*self.num_classes).to(self.device)
        imgs    = self.imgs(indices)

        basis = self.basis(torch.arange(self.n_basis).to(self.device))

        basis = basis.view(
                    self.n_basis,
                    self.channel,
                    self.im_size[0],
                    self.im_size[1]
                ).contiguous()

        imgs  = self.combine_basis(basis, imgs)
        return imgs.min().item(), imgs.max().item()

    def forward(self, placeholder=None, task_indices=None, combine=False, new_batch_size=None):
        
        combine=True
        
        if task_indices is None:
            task_indices = list(range(self.num_classes))
        assert isinstance(task_indices, list)#这里应该是外层优化每次只选取几个task 如果没指定就用全部的task
        indices = []
        for i in task_indices:
            if new_batch_size is None:
                ind = torch.randperm(self.ipc)[:self.n_per_c].sort()[0] + self.ipc * i
            else:
                ind = torch.randperm(self.ipc)[:new_batch_size].sort()[0] + self.ipc * i
            indices.append(ind)
        # 实际上用到的每个类别的数据个数self.n_per_c可能小于self.ipc 所以需要随机选取？
        
        basis = self.basis(torch.arange(self.n_basis).to(self.device))

        basis = basis.view(
                    self.n_basis,
                    self.channel,
                    self.im_size[0],
                    self.im_size[1]
                ).contiguous()#sb吧这不是前面的get_basis函数吗？

        indices = torch.cat(indices).to(self.device)
        imgs    = self.imgs(indices)
        labs    = self.labs[indices]

        if combine:
            imgs = self.combine_basis(basis, imgs)
            return imgs, labs, indices
        else:
            return [basis, imgs], labs, indices
