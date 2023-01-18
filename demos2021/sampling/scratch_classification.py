"""Python Script Template."""
from torch.distributions import Laplace, Normal
import torch

from data import FastMNIST
from training import (
    GPLearner,
    SGDLearner,
    BootstrapEnsembleLearner,
    EnsembleLearner,
    DropoutLearner,
    SGLDLearner,
    BayesBackPropLearner,
    SWAGLearner,
    MALALearner
)

import matplotlib.pyplot as plt
seed = 42

train_set = FastMNIST('../data', train=True, download=True)
test_set = FastMNIST('../data', train=False, download=True)

def train(learner, num_epochs, train_set, test_set, num_samples=1, title=""):
    losses = learner.train(num_epochs=num_epochs)
    plt.plot(losses)
    plt.show()
    return losses


learner = SGDLearner(
    train_set=train_set,
    batch_size=256,
    lr=1.0,
    weight_decay=1e-4,
    train_device="cpu",
)
losses = train(learner, 25, train_set, test_set, num_samples=1, title="SGD")