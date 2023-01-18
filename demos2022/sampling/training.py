"""Python Script Template."""
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.distributions import Poisson
from collections import deque
from tqdm import tqdm
import copy
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from sampling.models import RegressionNet, MNISTNet, ExactGP, BBP_Model_Wrapper, BBP_Heteroscedastic_Model
    from sampling.optimizers import SGLD, AdamLD, PSGLD
except ModuleNotFoundError:
    from models import RegressionNet, MNISTNet, ExactGP
    from optimizers import SGLD, AdamLD, PSGLD

from torch.optim import SGD, Adam
import torch
import torch.nn as nn
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions import Categorical, Normal




def reduce_predictions(out, deterministic=False):
    """Reduce predictions."""
    out_mean = torch.stack([o[0] for o in out])
    mean = torch.mean(out_mean, dim=0)
    epistemic_std = torch.std(out_mean, dim=0)

    if deterministic:
        epistemic_lower, epistemic_upper = None, None
        all_lower = mean - 2 * epistemic_std
        all_upper = mean + 2 * epistemic_std
    else:
        out_std = torch.stack([o[1] for o in out])
        aleatoric_std = torch.mean(out_std, dim=0)

        all_lower = mean - 2 * torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
        all_upper = mean + 2 * torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
        epistemic_lower = mean - 2 * epistemic_std
        epistemic_upper = mean + 2 * epistemic_std

    return mean, all_lower, all_upper, epistemic_lower, epistemic_upper


class Trainer(object):
    def __init__(self, train_set, batch_size, train_device="cpu", test_device="cpu"):
        self.train_set = train_set
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.num_batches = len(train_set) // batch_size
        self.train_device = train_device
        self.test_device = test_device

    def train(self, num_epochs):
        losses = []
        for _ in tqdm(range(num_epochs)):
            for data in self.train_loader:
                loss = self.fit(*data)
                losses.append(loss.to("cpu").item())
        return losses

    def fit(self, *args):
        raise NotImplementedError

    def evaluate(self, test_set, num_samples=1):
        raise NotImplementedError

    def predict(self, x, num_samples=1):
        """Output predictive distribution."""
        pass

    def sample(self, x):
        """Sample a function at values x."""
        pass

    def predict_normal(self, x, num_samples=1):
        """Predict normal distribution."""
        mean, all_lower, all_upper, epi_lower, epi_upper = self.predict(x)
        if epi_lower is not None:
            scale = (epi_upper - epi_lower) / 2
        elif all_upper is None:
            scale = torch.ones_like(mean)
        else:
            scale = (all_upper - all_lower) / 2
        return Normal(mean, scale)


class GPLearner(Trainer):
    def __init__(self, train_set, lr, momentum=0.1, *args, **kwargs):
        super().__init__(train_set, batch_size=len(train_set), *args, **kwargs)
        self.model = ExactGP(
                train_set.tensors[1].squeeze(), train_set.tensors[2].squeeze()
        )
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def fit(self, *args):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.model.train_inputs[0])
        loss = -self.mll(output, self.model.train_targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, x, num_samples=1):
        """Output predictive distribution."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            epistemic_lower, epistemic_upper = out.confidence_region()
            all_lower, all_upper = self.model.likelihood(out).confidence_region()
        return out.mean, all_lower, all_upper, epistemic_lower, epistemic_upper


class SGDLearner(Trainer):
    def __init__(self, train_set, batch_size, lr, momentum=0.0, weight_decay=0.0,
                 deterministic=False, dropout_p=0, dropout_at_eval=False,
                 base_layer=nn.Linear, prior=None, non_linearity='relu', *args, **kwargs):
        super().__init__(train_set, batch_size, *args, **kwargs)
        if isinstance(train_set, MNIST):
            self.model = MNISTNet(
                    dropout_at_eval=dropout_at_eval,
                    linear_layer=base_layer,
                    prior=prior
            ).to(self.train_device)
            self.task = "classification"
            self.distribution = Categorical
        else:
            self.model = RegressionNet(
                    dropout_p=dropout_p,
                    dropout_at_eval=dropout_at_eval,
                    base_layer=base_layer,
                    prior=prior,
                    non_linearity=non_linearity
            ).to(self.train_device)
            self.task = "regression"
            self.distribution = Normal

        self.deterministic = deterministic
            
        self.optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
        )

    def fit(self, idx, x, y, weight=1):
        """Fit a batch of data."""
        x, y = x.to(self.train_device), y.to(self.train_device)
        if isinstance(weight, torch.Tensor):
            weight = weight.to(self.train_device)

        self.optimizer.zero_grad()
        if self.deterministic and self.task == "regression":
            y_pred = self.model(x)[0]
            loss = (weight * ((y - y_pred) ** 2)).mean()
        else:
            loss = -(weight * self.distribution(*self.model(x)).log_prob(y)).mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, test_set, num_samples=1):
        self.model.to("cpu")
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        extra = 0
        log_lik = 0
        for (idx, x, y) in test_loader:
            predictive_distribution = self.distribution(*self.model(x))

            if self.deterministic and self.task == "regression":
                y_pred = predictive_distribution.mean
                mse = ((y - y_pred) ** 2).mean()
                log_lik += mse
            elif self.deterministic and self.task == "classification":
                log_lik += predictive_distribution.log_prob(y).mean()
                y_pred = predictive_distribution.logits.argmax()
            else:
                log_lik += predictive_distribution.log_prob(y).mean()
                y_pred = predictive_distribution.sample()

            if self.task == "classification":
                extra += (y == y_pred).float().mean()
            else:
                extra += ((y - y_pred) ** 2).mean()

        return log_lik, extra

    def predict(self, x, num_samples=1):
        """Output predictive distribution."""
        self.model.to(x.device)
        out = self.distribution(*self.model(x))
        mean, std2 = out.mean, 2 * out.stddev
        if self.deterministic:
            return mean, None, None, None, None
        else:
            return mean, mean.sub(std2), mean.add(std2), None, None

    def save_model(self, name, device="cpu"):
        self.model.to(device)
        torch.save(self.model.state_dict(), name + '.pt')

    def load_model(self, name, device="cpu"):
        self.model.load_state_dict(torch.load(name + '.pt'))
        self.model.to(device)

    def sample(self, x):
        """Sample a function at values x."""
        raise NotImplementedError


class EnsembleLearner(Trainer):
    def __init__(self, train_set, batch_size, num_heads, lr, momentum=0.0,
                 weight_decay=0.0,
                 deterministic=False,
                 non_linearity='relu',
                 *args, **kwargs):
        super().__init__(train_set, batch_size, *args, **kwargs)
        self.learners = [
            SGDLearner(
                    train_set,
                    batch_size,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    deterministic=deterministic,
                    non_linearity=non_linearity,
                    *args, **kwargs
            )
            for _ in range(num_heads)
        ]

    def train(self, num_epochs):
        for learner in self.learners:
            learner.train(num_epochs)

    def predict(self, x, num_samples=1):
        """Output predictive distribution."""
        out = []
        with torch.no_grad():
            for learner in self.learners:
                learner.model.to(x.device)
                out.append(learner.model(x))
        return reduce_predictions(out, deterministic=self.learners[0].deterministic)

    def save_model(self, name, device="cpu"):
        for i, learner in enumerate(self.learners):
            learner.save_model(f"{name}_{i}", device)

    def load_model(self, name, device="cpu"):
        for i, learner in enumerate(self.learners):
            learner.load_model(f"{name}_{i}", device)

    def sample(self, x):
        """Sample a function at values x."""
        learner = np.random.choice(self.learners)
        return learner.distribution(*learner.model(x)).mean


class BootstrapEnsembleLearner(Trainer):
    def __init__(self, train_set, batch_size, num_heads, lr, momentum=0.0,
                 weight_decay=0.0,
                 deterministic=False,
                 non_linearity='relu',
                 *args, **kwargs):
        super().__init__(train_set, batch_size, *args, **kwargs)
        self.num_heads = num_heads
        self.weights = Poisson(rate=1).sample((num_heads, len(train_set)))
        self.learners = [
            SGDLearner(
                    train_set,
                    batch_size,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    deterministic=deterministic,
                    non_linearity=non_linearity,
                    *args, **kwargs
            )
            for _ in range(num_heads)
        ]

    def fit(self, idx, x, y):
        """Fit a batch of data."""
        loss = 0
        for i, learner in enumerate(self.learners):
            loss += learner.fit(idx, x, y, weight=self.weights[i, idx].unsqueeze(-1))
        return loss / len(self.learners)

    def predict(self, x, num_samples=1):
        """Output predictive distribution."""
        out = []
        with torch.no_grad():
            for learner in self.learners:
                model = learner.model.to("cpu")
                out.append(model(x))
        return reduce_predictions(out, deterministic=self.learners[0].deterministic)

    def save_model(self, name, device="cpu"):
        for i, learner in enumerate(self.learners):
            learner.save_model(f"{name}_{i}", device)

    def load_model(self, name, device="cpu"):
        for i, learner in enumerate(self.learners):
            learner.load_model(f"{name}_{i}", device)

    def update_posterior(self):
        new_weight = Poisson(rate=1).sample((self.num_heads, 1))
        self.weights = torch.cat((self.weights, new_weight), -1)

    def sample(self, x):
        """Sample a function at values x."""
        learner = np.random.choice(self.learners)
        return learner.distribution(*learner.model(x)).mean


class DropoutLearner(SGDLearner):
    def __init__(self, dropout_p=0.5, dropout_at_eval=True, *args, **kwargs):
        super().__init__(
                dropout_p=dropout_p, dropout_at_eval=dropout_at_eval, *args, **kwargs
        )

    def predict(self, x, num_samples=200):
        """Output predictive distribution."""
        with torch.no_grad():
            self.model.to(x.device)
            out = [self.model(x) for _ in range(num_samples)]
        return reduce_predictions(out, deterministic=self.deterministic)

    def sample(self, x):
        """Sample a function at values x."""
        return self.distribution(*self.model(x)).mean


class SGLDLearner(SGDLearner):
    def __init__(
            self, lr,weight_decay, burn_in=50, sub_sample=50, max_size=30, num_iter=1000,
            *args, **kwargs
    ):
        super().__init__(lr=lr, *args, **kwargs)
        self.optimizer = SGLD(
                self.model.parameters(),
                lr=lr,
                weight_decay = weight_decay
                # **self.optimizer.defaults
        )
        self.models = deque()
        self.burn_in = burn_in
        self.sub_sample = sub_sample
        self.max_size = max_size

    def train(self, num_epochs):
        losses = []
        num_iter = 0
        loss = 0
        for _ in tqdm(range(num_epochs),desc=str(loss)):
            for data in self.train_loader:
                num_iter += 1
                loss = self.fit(*data)
                losses.append(loss.item())
                if num_iter > self.burn_in and num_iter % self.sub_sample == 0:
                    self.models.append(copy.deepcopy(self.model))
                if len(self.models) > self.max_size:
                    self.models.popleft()


        return losses

    def predict(self, x, num_samples=10):
        """Output predictive distribution."""
        out = []
        with torch.no_grad():
            for model in self.models:
                model.to(x.device)
                out.append(model(x))
        return reduce_predictions(out, deterministic=self.deterministic)

    def save_model(self, name, device="cpu"):
        for i, model in enumerate(self.models):
            model.to(device)
            torch.save(model.state_dict(), f"{name}_{i}.pt")

    def load_model(self, name, device="cpu"):
        i = 0
        model_available = True
        while model_available:
            try:
                self.models.append(copy.deepcopy(self.model))
                self.models[-1].load_state_dict(torch.load(f"{name}_{i}.pt"))
                self.models[-1].to(device)
                i = i + 1
            except FileNotFoundError:
                self.models.pop()
                model_available = False

    def sample(self, x):
        """Sample a function at values x."""
        model = np.random.choice(self.models)
        return self.distribution(*model(x)).mean


class MALALearner(SGLDLearner):
    def train(self, num_epochs):
        losses = []
        num_iter = 0
        for _ in tqdm(range(num_epochs)):
            for data in self.train_loader:
                num_iter += 1
                old_model = copy.deepcopy(self.model)
                accepted = False
                while not accepted:
                    loss = self.fit(*data)

                    x, y = data[1:]
                    x, y = x.to(self.train_device), y.to(self.train_device)
                    new_loss = -self.distribution(*self.model(x)).log_prob(y).mean()

                    log_q0 = self._proposal_dist(old_model, self.model, x, y)
                    log_q1 = self._proposal_dist(self.model, old_model, x, y)

                    p1p0 = torch.exp(-new_loss + loss)
                    q0q1 = torch.exp(log_q0 - log_q1)

                    alpha = (p1p0 * q0q1).clamp_max_(1.0)
                    if torch.rand([1]) <= alpha.to("cpu"):  # Accept proposal
                        accepted = True
                    else:  # Reject proposal
                        for new, old in zip(
                                self.model.parameters(), old_model.parameters()
                        ):
                            new.data = old.data

                losses.append(loss.item())
                if num_iter > self.burn_in and num_iter % self.sub_sample == 0:
                    self.models.append(copy.deepcopy(self.model))
                if len(self.models) > self.max_size:
                    self.models.popleft()


        return losses

    def _proposal_dist(self, new_model, old_model, x, y):
        """Get log_q(x'|x)."""
        tau = self.optimizer.param_groups[0]["lr"]
        out = 0

        for old in old_model.parameters():
            old.requires_grad = True
            if old.grad is not None:
                old.grad = torch.zeros_like(old.grad)

        loss = -self.distribution(*old_model(x)).log_prob(y).mean()
        loss.backward()
        for new, old in zip(new_model.parameters(), old_model.parameters()):
            out += -torch.norm((new - old - tau * old.grad)) ** 2

        return out


class SWAGLearner(SGDLearner):
    """SWAG."""

    def __init__(self, burn_in=5, sub_sample=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_model = copy.deepcopy(self.model)
        self.mean_model.to(self.train_device)
        self.squared_model = copy.deepcopy(self.model)
        self.squared_model.to(self.train_device)

        self.burn_in = burn_in
        self.sub_sample = sub_sample
        self.count = 0

    def train(self, num_epochs):
        losses = []
        for i_epoch in tqdm(range(num_epochs)):
            for data in self.train_loader:
                loss = self.fit(*data)
                losses.append(loss.item())
            if i_epoch > self.burn_in and i_epoch % self.sub_sample == 0:
                for running, mean, square in zip(
                        self.model.parameters(),
                        self.mean_model.parameters(),
                        self.squared_model.parameters()
                ):
                    n = self.count
                    mean.data = (n * mean.data + running.data) / (n + 1)
                    square.data = (n * square.data + running.data ** 2) / (n + 1)
                self.count += 1

        return losses

    def sample_model(self, device):
        """Sample a model."""
        sample_model = copy.deepcopy(self.model)
        sample_model = sample_model.to(device)
        for sample, mean, square in zip(
                sample_model.parameters(),
                self.mean_model.parameters(),
                self.squared_model.parameters()
        ):
            sample.data = Normal(mean.to(device),
                                 torch.sqrt(square - mean ** 2).to(device)
                                 ).sample()

        return sample_model

    def predict(self, x, num_samples=10):
        """Output predictive distribution."""
        with torch.no_grad():
            out = [self.sample_model(x.device)(x) for _ in range(num_samples)]
        return reduce_predictions(out, deterministic=self.deterministic)

    def save_model(self, name, device="cpu"):
        mean_model = self.mean_model.to(device)
        torch.save(mean_model.state_dict(), f"{name}_mean.pt")

        squared_model = self.squared_model.to(device)
        torch.save(squared_model.state_dict(), f"{name}_squared.pt")

    def load_model(self, name, device="cpu"):
        self.mean_model.load_state_dict(torch.load(f"{name}_mean.pt"))
        self.squared_model.load_state_dict(torch.load(f"{name}_squared.pt"))
        self.mean_model.to(device)
        self.squared_model.to(device)

    def sample(self, x):
        """Sample a function at values x."""
        return self.distribution(*self.sample_model(x.device)(x)).mean


class BayesBackPropLearner(Trainer):
    def __init__(self, train_set, batch_size, lr, momentum=0.0, weight_decay=0.0,
                 deterministic=False, dropout_p=0, dropout_at_eval=False,
                 base_layer=nn.Linear, prior=None, non_linearity='relu', *args, **kwargs):
        super().__init__(train_set, batch_size, *args, **kwargs)

        self.model = BBP_Model_Wrapper(network=BBP_Heteroscedastic_Model(input_dim=1, output_dim=1, num_units=200),
                                                learn_rate=1e-2, batch_size=batch_size, no_batches=1)
        self.task = "regression"
        self.distribution = Normal
        self.train_device =  "cpu"

        self.deterministic = deterministic
            

    def train(self, num_epochs):
        losses = []
        fit_loss_train = np.zeros(num_epochs)
        KL_loss_train = np.zeros(num_epochs)
        total_loss = np.zeros(num_epochs)
        best_net, best_loss = None, float('inf')

        for i in tqdm(range(num_epochs)):
            for data in self.train_loader:
                fit_loss, KL_loss = self.fit(*data)
                fit_loss_train[i] += fit_loss.cpu().data.numpy()
                KL_loss_train[i] += KL_loss.cpu().data.numpy()
                total_loss[i] = fit_loss_train[i] + KL_loss_train[i]

                if fit_loss < best_loss:
                    best_loss = fit_loss
                    best_net = copy.deepcopy(self.model.network)
                    self.best_net = best_net

        return total_loss


    def fit(self, idx, x, y, weight=1):
        """Fit a batch of data."""
        x, y = x.to(self.train_device), y.to(self.train_device)

        if isinstance(weight, torch.Tensor):
            weight = weight.to(self.train_device)

        fit_loss, KL_loss = self.model.fit(x, y, no_samples = 10)
        return  fit_loss, KL_loss

    def evaluate(self, test_set, num_samples=1):
        self.model.to("cpu")
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        extra = 0
        log_lik = 0
        for (idx, x, y) in test_loader:
            predictive_distribution = self.distribution(*self.model(x))

            if self.deterministic and self.task == "regression":
                y_pred = predictive_distribution.mean
                mse = ((y - y_pred) ** 2).mean()
                log_lik += mse
            elif self.deterministic and self.task == "classification":
                log_lik += predictive_distribution.log_prob(y).mean()
                y_pred = predictive_distribution.logits.argmax()
            else:
                log_lik += predictive_distribution.log_prob(y).mean()
                y_pred = predictive_distribution.sample()

            if self.task == "classification":
                extra += (y == y_pred).float().mean()
            else:
                extra += ((y - y_pred) ** 2).mean()

        return log_lik, extra

    def predict(self, x, num_samples=10):
        """Output predictive distribution."""
        mean_list = []
        noise_list = []
        for _ in range(num_samples):
            preds = self.best_net.forward(x)[0]
            mean_list.append(preds.cpu().data[:, 0])
            noise_list.append(preds[:, 1].exp().cpu().data)

        out_mean = torch.stack(mean_list)
        mean = torch.mean(out_mean, dim=0)
        epistemic_std = torch.std(out_mean, dim=0)

        if self.deterministic:
            epistemic_lower, epistemic_upper = None, None
            all_lower = mean - 2 * epistemic_std
            all_upper = mean + 2 * epistemic_std
        else:
            out_std = torch.stack(noise_list)
            aleatoric_std = torch.mean(out_std, dim=0)

            all_lower = mean - 2 * torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
            all_upper = mean + 2 * torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
            epistemic_lower = mean - 2 * epistemic_std
            epistemic_upper = mean + 2 * epistemic_std
        
        return mean, all_lower, all_upper, epistemic_lower, epistemic_upper

    def save_model(self, name, device="cpu"):
        self.best_net.to(device)
        torch.save(self.best_net.state_dict(), name + '.pt')

    def load_model(self, name, device="cpu"):
        self.model.load_state_dict(torch.load(name + '.pt'))
        self.model.to(device)

    def sample(self, x):
        """Sample a function at values x."""
        raise NotImplementedError


class FsvgdLearner(Trainer):
    def __init__(self, train_set, batch_size, lr, num_heads=5, momentum=0.0, weight_decay=0.0,
                 dropout_p=0, dropout_at_eval=False,
                 base_layer=nn.Linear, prior=None, non_linearity='relu',
                 domain_l=-3, domain_u=3., mset_size=5, svgd_bandwidth=1.0, likelihood_std=0.1,
                 *args, **kwargs):
        super().__init__(train_set, batch_size, *args, **kwargs)
        assert not isinstance(train_set, MNIST)

        self.num_heads = num_heads
        self.models = nn.ModuleList([RegressionNet(
            dropout_p=dropout_p,
            dropout_at_eval=dropout_at_eval,
            base_layer=base_layer,
            prior=prior,
            non_linearity=non_linearity
        ).to(self.train_device) for _ in range(num_heads)])
        self.task = "regression"
        self.distribution = Normal
        self.domain_bounds = (domain_l, domain_u)
        self.mset_size = mset_size
        self.svgd_bandwidth = svgd_bandwidth
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size

        self.optimizer = Adam(
            self.models.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def forward(self, x):
        means_stacked = torch.stack([model(x)[0] for model in self.models], dim=0)
        scales_stacked = torch.stack([model(x)[1] for model in self.models], dim=0)
        return means_stacked, scales_stacked

    def fit(self, idx, x, y, weight=1):
        """Fit a batch of data."""
        x, y = x.to(self.train_device), y.to(self.train_device)
        assert y.shape[-1] == 1, 'y must be 1-dimensional'
        y = y.reshape((-1,))

        if isinstance(weight, torch.Tensor):
            weight = weight.to(self.train_device)

        assert self.task == "regression"

        self.optimizer.zero_grad()

        # sample batch
        N = x.shape[0]
        batch_idx = torch.randperm(N)[:self.batch_size]
        x_batch, y_batch = x[batch_idx, :], y[batch_idx]

        # sample measurement sets
        x_mset = torch.distributions.Uniform(*self.domain_bounds).sample((self.mset_size, 1))

        # forward pass through the NNs
        x_cat = torch.cat([x_batch, x_mset], dim=0)
        pred_means = self.forward(x_cat)[0].squeeze()
        f_data, f_mset = pred_means[:, :-self.mset_size], pred_means[:, -self.mset_size:]

        # compute likelihood score
        loss = -(weight * self.distribution(f_data, self.likelihood_std).log_prob(y_batch)).sum() * (N / self.batch_size)
        likelihood_score = torch.autograd.grad(loss, f_data)[0]

        # compute GP prior score
        K_gp_prior = torch.exp(- torch.cdist(x_mset, x_mset.detach()) / (2.))
        prior_dist = torch.distributions.MultivariateNormal(loc=torch.zeros_like(f_mset), covariance_matrix=K_gp_prior)
        prior_score = - torch.autograd.grad(prior_dist.log_prob(f_mset).sum(), f_mset)[0]

        # concatenate the scores
        score = torch.cat([likelihood_score, prior_score], dim=-1)

        # compute SVGD update in the function space
        K_svgd = torch.exp(- torch.cdist(pred_means, pred_means.detach()) / (2 * self.svgd_bandwidth))
        grad_K = torch.autograd.grad(K_svgd.sum(), pred_means)[0]
        svgd_update_f = K_svgd.matmul(score) + grad_K

        # construct surrogate fSCGD objective so that its gradients correspond to the fSVGD updates of the NN particles
        svgd_surrogate = (pred_means * (svgd_update_f).detach()).sum()
        svgd_surrogate.backward()

        self.optimizer.step()
        return loss

    def evaluate(self, test_set, num_samples=1):
        self.model.to("cpu")
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        extra = 0
        log_lik = 0
        for (idx, x, y) in test_loader:
            predictive_distribution = self.distribution(*self.model(x))

            if self.deterministic and self.task == "regression":
                y_pred = predictive_distribution.mean
                mse = ((y - y_pred) ** 2).mean()
                log_lik += mse
            elif self.deterministic and self.task == "classification":
                log_lik += predictive_distribution.log_prob(y).mean()
                y_pred = predictive_distribution.logits.argmax()
            else:
                log_lik += predictive_distribution.log_prob(y).mean()
                y_pred = predictive_distribution.sample()

            if self.task == "classification":
                extra += (y == y_pred).float().mean()
            else:
                extra += ((y - y_pred) ** 2).mean()

        return log_lik, extra

    def predict(self, x, num_samples=1):
        """Output predictive distribution."""
        out = []
        with torch.no_grad():
            for model in self.models:
                model.to(x.device)
                out.append(model(x))
        return reduce_predictions(out, deterministic=True)

    def save_model(self, name, device="cpu"):
        self.models.to(device)
        torch.save(self.models.state_dict(), name + '.pt')

    def load_model(self, name, device="cpu"):
        self.models.load_state_dict(torch.load(name + '.pt'))
        self.models.to(device)

    def sample(self, x):
        """Sample a function at values x."""
        """Sample a function at values x."""
        model = np.random.choice(self.models)
        return model(x)[0]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    def plot_predictions(true_x, true_y, train_set, test_set, learner, num_samples=1, title=""):
        idx = true_x.sort()[1]
        eval_x = torch.linspace(-3, 3, 100).unsqueeze(-1)
        plt.plot(true_x[idx], true_y[idx], 'b-', label="Sampled Function")
        plt.plot(train_set.tensors[1], train_set.tensors[2], 'b*', markersize=15, label="Train data")
        #     plt.plot(test_set.tensors[1], test_set.tensors[2], 'k*', markersize=15, label="Test data")
        with torch.no_grad():
            mean, lower, upper, epistemic_lower, epistemic_upper = learner.predict(
                eval_x, num_samples=num_samples
            )
        plt.plot(eval_x.squeeze(), mean.numpy(), "g-", label="Mean Prediction")

        if lower is None and epistemic_lower is None:
            pass
        elif epistemic_lower is None:
            plt.fill_between(
                eval_x.squeeze().numpy(), lower.squeeze().numpy(), upper.squeeze().numpy(),
                color="g", alpha=0.3, label="Total Uncertainty"
            )
        else:

            plt.fill_between(
                eval_x.squeeze().numpy(),
                epistemic_lower.squeeze().numpy(),
                epistemic_upper.squeeze().numpy(),
                color="g", alpha=0.3, label="Epistemic Uncertainty"
            )

            plt.fill_between(
                eval_x.squeeze().numpy(),
                epistemic_upper.squeeze().numpy(),
                upper.squeeze().numpy(),
                color="b", alpha=0.3, label="Aleatoric Uncertainty"
            )
            plt.fill_between(
                eval_x.squeeze().numpy(),
                lower.squeeze().numpy(),
                epistemic_lower.squeeze().numpy(),
                color="b", alpha=0.3
            )
        plt.legend(frameon=False, ncol=3)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title(title)
        plt.show()

    from sampling.data import heteroskedastic_regression, homoskedastic_regression

    print('cde')

    seed = 42
    (homo_train_set, homo_test_set), true_x, true_y, kernel = homoskedastic_regression(noise=0.02,
        num_points=3, seed=seed)

    train_set, test_set = homo_train_set, homo_test_set

    learner = FsvgdLearner(
            train_set, batch_size=len(train_set),
            num_heads=10, lr=1e-3, weight_decay=1e-2, train_device='cpu',
            svgd_bandwidth=0.2
    )
    learner.train(num_epochs=1000)

    plot_predictions(
            true_x, true_y, train_set, test_set, learner, num_samples=5, title='Ensemble'
    )