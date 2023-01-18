"""Python Script Template."""
import torch.nn as nn
import torch
import gpytorch
from torch.distributions import Normal, Laplace, kl_divergence
import torch.nn.functional as func

JITTER = 1e-6


class BayesianLayer(nn.Module):
    def __init__(self, base_layer=nn.Linear, prior=Normal(0, 0.1), *args, **kwargs):
        super().__init__()
        self.mean = base_layer(*args, **kwargs)
        self.scale = base_layer(*args, **kwargs)
        
        self.linear = base_layer == nn.Linear
        self.prior = prior
        self.num_samples = 10
        self._sample_weights()

    def _sample_weights(self):
        self.weight_distribution = Normal(
                self.mean.weight,
                func.softplus(self.scale.weight, beta=1, threshold=20) + JITTER
        )
        self.bias_distribution = Normal(
                self.mean.bias,
                func.softplus(self.scale.bias, beta=1, threshold=20) + JITTER
        )
        weight = self.weight_distribution.rsample((self.num_samples,)).mean(0)
        bias = self.bias_distribution.rsample((self.num_samples,)).mean(0)
        return weight, bias

    def kl(self):
        """Return kl div between weights and prior."""
        kl = kl_divergence(self.prior, self.weight_distribution).mean()
        kl += kl_divergence(self.prior, self.bias_distribution).mean()
        return kl / 2

    def forward(self, x):
        weight, bias = self._sample_weights()
        return func.linear(x, weight, bias)


class RegressionNet(nn.Module):
    def __init__(
            self,
            in_dim=1,
            out_dim=1,
            dropout_p=0.5,
            dropout_at_eval=False,
            base_layer=nn.Linear,
            prior=None,
            non_linearity='relu'
    ):
        super().__init__()
        
        if prior is not None:
            self.w1 = BayesianLayer(in_features=in_dim, out_features=128, prior=prior)
            self.w2 = BayesianLayer(in_features=128, out_features=64, prior=prior)
            self.head = BayesianLayer(in_features=64, out_features=out_dim, prior=prior)
            self.scale_head = BayesianLayer(in_features=64, out_features=out_dim, prior=prior)
        else:
            self.w1 = nn.Linear(in_features=in_dim, out_features=128)
            self.w2 = nn.Linear(in_features=128, out_features=64)
            self.head = nn.Linear(in_features=64, out_features=out_dim)
            self.scale_head = nn.Linear(in_features=64, out_features=out_dim)
            
        self.dropout_p = dropout_p
        self.dropout_at_eval = dropout_at_eval
        if non_linearity == 'relu':
            self.activation = torch.nn.functional.relu
        else:
            self.activation = torch.tanh

    def kl(self):
        """Compute log-likelihood and prior of weights."""
        return self.w1.kl() + self.w2.kl() + self.head.kl() + self.scale_head.kl()

    def forward(self, x):
        h1 = func.dropout(
                self.activation(self.w1(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )
        h2 = func.dropout(
                self.activation(self.w2(h1)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )

        mean = self.head(h2)
        scale = func.softplus(self.scale_head(h2)) + 1e-2

        return mean, scale


class ExactGP(gpytorch.models.ExactGP):
    def __init__(
            self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
    ):
        super().__init__(train_x, train_y, likelihood=likelihood)
        self.likelihood.noise = 0.01
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MNISTNet(nn.Module):
    def __init__(self,
                 dropout_p=0.5,
                 dropout_at_eval=False,
                 linear_layer=nn.Linear,
                 prior=None,
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = linear_layer(9216, 128)
        self.fc2 = linear_layer(128, 10)
        self.dropout_p = dropout_p
        self.dropout_at_eval = dropout_at_eval

    def forward(self, x):
        x = func.dropout(
                func.relu(self.conv1(x)),
                p=self.dropout_p / 2,
                training=self.training or self.dropout_at_eval
        )
        x = func.dropout(
                func.max_pool2d(func.relu(self.conv2(x)), 2),
                p=self.dropout_p / 2,
                training=self.training or self.dropout_at_eval
        )
        x = torch.flatten(x, 1)
        x = func.dropout(
                func.relu(self.fc1(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )

        class_probs = self.fc2(x)
        return class_probs,


