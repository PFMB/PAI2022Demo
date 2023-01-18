"""Python Script Template."""
import torch.nn as nn
import torch
import gpytorch
from torch.distributions import Normal, Laplace, kl_divergence
import torch.nn.functional as func
import numpy as np


from torch.autograd import Variable

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
    if sum_reduce:
        return -(log_coeff + exponent).sum()
    else:
        return -(log_coeff + exponent)


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)
    
    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()
    
    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()


class BayesianLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesianLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        
        scale = (2/self.input_dim)**0.5
        rho_init = np.log(np.exp((2/self.input_dim)**0.5) - 1)
        self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3))
        
        self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -3))
        
    def forward(self, x, sample = True):
        
        if sample:
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            bias_epsilons =  Variable(self.bias_mus.data.new(self.bias_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            bias_sample = self.bias_mus + bias_epsilons*bias_stds
            
            output = torch.mm(x, weight_sample) + bias_sample
            
            # computing the KL loss term
            prior_cov, varpost_cov = self.prior.sigma**2, weight_stds**2
            KL_loss = 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*weight_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.weight_mus - self.prior.mu)**2/prior_cov).sum()
            
            prior_cov, varpost_cov = self.prior.sigma**2, bias_stds**2
            KL_loss = KL_loss + 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*bias_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.bias_mus - self.prior.mu)**2/prior_cov).sum()
            
            return output, KL_loss
        
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, KL_loss
        
    def sample_layer(self, no_samples):
        all_samples = []
        for i in range(no_samples):
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            
            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()
            
        return all_samples


class BBP_Heteroscedastic_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(BBP_Heteroscedastic_Model, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # network with two hidden and one output layer
        self.layer1 = BayesianLayer(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesianLayer(num_units, 2*output_dim, gaussian(0, 1))
        
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace = True)
    
    def forward(self, x):
        
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)
        
        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)
        
        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss
        
        return x, KL_loss_total


class BBP_Model_Wrapper:
    def __init__(self, network, learn_rate, batch_size, no_batches):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches
        
        self.network = network
        # self.network.cuda()
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learn_rate)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=False)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0
        
        for i in range(no_samples):
            output, KL_loss_total = self.network(x)

            # calculate fit loss based on mean and standard deviation of output
            fit_loss = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1)
            fit_loss_total = fit_loss_total + fit_loss
        
        KL_loss_total = KL_loss_total/self.no_batches
        total_loss = (fit_loss_total + KL_loss_total)/(no_samples*x.shape[0])
        total_loss.backward()
        self.optimizer.step()

        return fit_loss_total/no_samples, KL_loss_total
    
    def get_loss_and_rmse(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=False)
        
        means, stds = [], []
        for i in range(no_samples):
            output, KL_loss_total = self.network(x)
            means.append(output[:, :1, None])
            stds.append(output[:, 1:, None].exp())
            
        means, stds = torch.cat(means, 2), torch.cat(stds, 2)
        mean = means.mean(dim=2)
        std = (means.var(dim=2) + stds.mean(dim=2)**2)**0.5
            
        # calculate fit loss based on mean and standard deviation of output
        logliks = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1, sum_reduce=False)
        rmse = float((((mean - y)**2).mean()**0.5).cpu().data)

        return logliks, rmse



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
        scale = torch.exp(self.scale_head(h2))
        scale = torch.sqrt(scale)

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


