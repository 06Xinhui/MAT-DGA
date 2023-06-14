import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


# class DyReLUA(DyReLU):
#     def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
#         super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
#         self.fc2 = nn.Linear(channels // reduction, 2*k)

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         theta = self.get_relu_coefs(x)

#         relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
#         # BxCxL -> LxCxBx1
#         x_perm = x.transpose(0, -1).unsqueeze(-1)
#         output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
#         # LxCxBx2 -> BxCxL
#         result = torch.max(output, dim=-1)[0].transpose(0, -1)

#         return result



class DyReLUA(nn.Module):
    def __init__(self, channels, reduction=1, k=2, lambdas=None, init_values=None):
        super(DyReLUA, self).__init__()

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels//reduction, 2*k)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Defining lambdas in form of [La1, La2, Lb1, Lb2]
        if lambdas is not None:
            self.lambdas = lambdas
        else:
            # Default lambdas from DyReLU paper
            self.lambdas = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float).cuda()

        # Defining Initializing values in form of [alpha1, alpha2, Beta1, Beta2]
        if lambdas is not None:
            self.init_values = init_values
        else:
            # Default initializing values of DyReLU paper
            self.init_values = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float).cuda()

    def forward(self, F_tensor):

        # Global Averaging F
        kernel_size = F_tensor.shape[2:] # Getting HxW of F
        gap_output = F.avg_pool2d(F_tensor, kernel_size)

        # Flattening gap_output from (batch_size, C, 1, 1) to (batch_size, C)
        gap_output = gap_output.flatten(start_dim=1)

        # Passing Global Average output through Fully-Connected Layers
        x = self.relu(self.fc1(gap_output))
        x = self.fc2(x)
        
        # Normalization between (-1, 1)
        residuals = 2 * self.sigmoid(x) - 1

        # Getting values of theta, and separating alphas and betas
        theta = self.init_values + self.lambdas * residuals # Contains[alpha1(x), alpha2(x), Beta1(x), Beta2(x)]
        alphas = theta[0, :2]
        betas = theta[0, 2:]

        # Performing maximum on both piecewise functions
        output = torch.maximum((alphas[0] * F_tensor + betas[0]), (alphas[1] * F_tensor + betas[1]))

        return output

class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result.contiguous()