"""Python Script Template."""
import torch
from torch.optim import SGD
from tqdm import tqdm
from sampling.optimizers import PSGLD, SGLD


class Sampler(object):
    def __init__(
            self, x, func, preconditioned=False, noise_free=False, lr=1e-2,
            lr_final=1e-4, max_iter=1e4
    ):
        self.x = x
        self.func = func
        if preconditioned:
            optimizer = PSGLD
        else:
            optimizer = SGLD
        if noise_free:
            optimizer = SGD
        self.optimizer = optimizer([self.x], lr, weight_decay=0.0)

        if lr_final == lr:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lambda t: lr
            )
        else:
            gamma = -0.55
            b = max_iter / ((lr_final / lr) ** (1 / gamma) - 1.0)
            a = lr / (b ** gamma)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lambda t: a * ((b + t) ** gamma))

    def sample(self, max_iter, subsampling=3):
        """Sample"""
        samples = []
        losses = []
        p_accept = []
        for j in tqdm(range(max_iter)):
            sample, loss, alpha, _ = self.sample_next()
            if j % subsampling == 0:
                losses.append(loss.detach())
                samples.append(sample.cpu().numpy())
                p_accept.append(alpha)

        return samples, losses, p_accept

    def sample_next(self):
        """Sample a point."""
        raise NotImplementedError


class SGLDSampler(Sampler):

    def sample_next(self):
        """Sample a point."""
        self.optimizer.zero_grad()
        loss = -self.func(self.x)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return self.x.clone().detach(), loss.detach(), 1., self.x.clone().detach()


class MALASampler(Sampler):

    def sample_next(self):
        """Sample a point."""
        old_x = self.x.clone().detach()
        # while not accepted:
        self.optimizer.zero_grad()
        loss = -self.func(self.x)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        new_loss = -self.func(self.x)
        # Compute acceptance probability
        p1p0 = torch.exp(-new_loss + loss)
        q0q1 = torch.exp(
                self._proposal_dist(self.x, old_x) - self._proposal_dist(old_x, self.x)
        )
        alpha = (p1p0 * q0q1)
        if torch.rand([1]) <= alpha.clamp_max_(1.0):  # Accept proposal
            proposal = self.x.data.clone().detach()
            # accepted = True
        else:  # Reject proposal
            proposal = self.x.data.clone().detach()
            self.x.data = old_x.data

        self.scheduler.step()

        return self.x.clone().detach(), loss.detach(), alpha.item(), proposal

    def _proposal_dist(self, x, x_old):
        try:
            V = self.optimizer.state[
                self.optimizer.param_groups[0]["params"][0]
            ]["square_avg"]
            G = V.sqrt().add(self.optimizer.param_groups[0]['eps'])
        except KeyError:
            G = 1

        tau = self.optimizer.param_groups[0]["lr"]
        if x_old.grad is not None:
            grad_loss = x_old.grad
        else:
            x_ = torch.zeros_like(x)
            x_.data = x_old.data
            x_.requires_grad = True
            loss = -self.func(x_)
            grad_loss = torch.autograd.grad(loss, x_, retain_graph=False)[0]
        return -torch.norm((x - x_old - tau * grad_loss) / G) ** 2
