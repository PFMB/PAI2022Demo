"""Python Script Template."""
import torch
from torch.optim import SGD, RMSprop, Adam


class SGLD(SGD):
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
                noise_std = torch.tensor(2 * group['lr']).sqrt()
                noise = torch.distributions.Normal(
                        torch.zeros_like(p.data),
                        scale=noise_std.to(p.data.device)
                ).sample()
                p.data.add_(noise)

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

