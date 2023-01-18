"""Python Script Template."""
import torch
from torch.optim import SGD, RMSprop, Adam
from sys import exit

class SGLD(SGD):
    """Implementation of SGLD algorithm.
    References
    ----------
        
    """
    @torch.no_grad()
    def step(self, closure=None):
        """See `torch.optim.step’."""
        loss = super().step(closure)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_p = p.grad.data
                if weight_decay!=0:
                    grad_p.add_(alpha=weight_decay,other=p.data)
                langevin_noise = torch.randn_like(p.data).mul_(group['lr']**0.5)*0.1 #  use weight 0.1 to balance the noise
                p.data.add_(grad_p,alpha=-0.5*group['lr'])
                if torch.isnan(p.data).any(): 
                    exit('Exist NaN param after SGLD, Try to tune the parameter')
                if torch.isinf(p.data).any(): 
                    exit('Exist Inf param after SGLD, Try to tune the parameter')
                p.data.add_(langevin_noise)
        return loss

class AdamLD(Adam):
    """Implementation of Adam-LD algorithm.

    References
    ----------
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    """

    @torch.no_grad()
    def step(self, closure=None):
        """See `torch.optim.step'."""
        loss = super().step(closure)

        for group in self.param_groups:
            for p in group['params']:
                noise_std = torch.tensor(2 * group['lr']).sqrt()
                noise = torch.distributions.Normal(
                        torch.zeros_like(p.data),
                        scale=noise_std.to(p.data.device)
                ).sample()
                p.data.add_(noise)

        return loss


class PSGLD(RMSprop):
    """Implementation of SGLD algorithm.

    References
    ----------
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    """

    @torch.no_grad()
    def step(self, closure=None):
        """See `torch.optim.step'."""
        loss = super().step(closure)

        for group in self.param_groups:
            for p in group['params']:
                V = self.state[p]['square_avg']
                G = V.sqrt().add(group['eps'])
                if torch.any(G < 10 * group['eps']):
                    noise_std = torch.tensor(2 * group['lr']).sqrt()
                else:
                    noise_std = (2 * group['lr'] / G).sqrt()
                noise = torch.distributions.Normal(
                        torch.zeros_like(p.data),
                        scale=noise_std.to(p.data.device)
                ).sample()
                p.data.add_(noise)

        return loss

