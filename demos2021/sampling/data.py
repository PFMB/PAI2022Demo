"""Dataset scripts."""
import torch
import gpytorch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import copy


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return index, img, target


class IndexMNIST(MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index, img, target


class RegressionDataset(TensorDataset):

    def __init__(self, *args):
        super().__init__(*args)
        self.all_tensors = copy.deepcopy(self.tensors)
        self._len = self.tensors[0].shape[0]

    @property
    def num_points(self):
        return self._len

    @num_points.setter
    def num_points(self, new_size):
        full_size = self.all_tensors[0].shape[0]
        new_size = min(new_size, full_size)
        self._len = new_size
        select_idx = np.random.choice(full_size, self._len, replace=False)
        self.tensors = list(tensor[select_idx] for tensor in self.all_tensors)
        self.tensors[0] = torch.arange(new_size)

    def __len__(self):
        return self.num_points


class ActiveLearningDataset(TensorDataset):
    def __init__(self, *args):
        super().__init__(*args)
        self._seen_indexes = []

    @property
    def x(self):
        return self.tensors[0]

    @property
    def y(self):
        return self.tensors[1]

    def get_observed_data(self):
        return self.x[self._seen_indexes], self.y[self._seen_indexes]

    def query_index(self, index):
        self._seen_indexes.append(index)

    def __len__(self):
        return len(self._seen_indexes)

    def __getitem__(self, item):
        with torch.no_grad():
            idx = self._seen_indexes[item]
            return item, self.x[idx].detach(), self.y[idx].detach()


def get_gp_sample(num_points=1000, seed=None):
    """Get a sample from GP model."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.rand(num_points) * 6 - 3

    mean_x = gpytorch.means.ZeroMean()(x)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    kernel.outputscale = 0.3
    kernel.base_kernel.lengthscale = 1.0

    covar_x = kernel(x)
    prior = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    y = prior.sample()
    return x, y, kernel


def split_into_train_test(x, y, num_points, train_fraction=0.8):
    """Split data into train/test."""
    train_points = int(train_fraction * num_points)

    train_set = RegressionDataset(
            torch.arange(len(y)),
            x.unsqueeze(-1),
            y.unsqueeze(-1)
    )

    test_set = RegressionDataset(
            torch.arange(len(y)),
            x.unsqueeze(-1),
            y.unsqueeze(-1)
    )
    train_set.num_points = num_points
    test_set.num_points = int(train_fraction * num_points)
    return train_set, test_set


def homoskedastic_regression(noise=0.1, num_points=200, seed=None):
    """Create a homoskedastic regression data loader."""
    np.random.seed(seed + 1)
    true_x, true_y, kernel = get_gp_sample(seed=seed)
    idx = torch.bitwise_and(true_x < 2, true_x > -2)
    x, y = true_x[idx], true_y[idx]
    y = y + noise * torch.randn(y.shape)
    return split_into_train_test(x, y, num_points=num_points), true_x, true_y, kernel


def heteroskedastic_regression(noise=0.1, num_points=200, seed=None):
    """Create a heteroskedastic regression data loader."""
    np.random.seed(seed + 1)
    true_x, true_y, kernel = get_gp_sample(seed=seed)
    idx = torch.bitwise_and(true_x < 2, true_x > -2)
    x, y = true_x[idx], true_y[idx]
    noise = noise * torch.abs(torch.sin(2 * x) + 0.1) * (0.5 + (1 + x))
    y = y + noise * torch.randn(y.shape)
    return split_into_train_test(x, y, num_points=num_points), true_x, true_y, kernel


def mnist_no_zero():
    """Create a mnist without a zero data loader."""
    train_set = FastMNIST('data/MNIST', train=True, download=True)
    not_zero_idx = train_set.targets != 0
    train_set.data = train_set.data[not_zero_idx]
    train_set.targets = train_set.targets[not_zero_idx]

    test_set = FastMNIST('data/MNIST', train=False, download=True)
    zero_idx = test_set.targets == 0
    test_set.data = test_set.data[zero_idx]
    test_set.targets = test_set.targets[zero_idx]
    return train_set, test_set


def rotated_mnist():
    """Create a rotated mnist dataset."""
    train_set = FastMNIST('data/MNIST', train=True, download=True)
    test_set = MNIST('data/MNIST', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.RandomRotation(degrees=5),
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])
                     )
    # Change with test_set.transforms.transform.transforms[0].degrees = (-15, 15)
    return train_set, test_set
