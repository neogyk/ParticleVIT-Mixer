"""
This is code adaptation of the MambaBlock - https://github.com/myscience/mamba/tree/main
Original Mamba Paper: https://arxiv.org/pdf/2312.00752
"""

import torch
import einops
from einops import einsum
from typing import Tuple, TypeVar
from torch import Tensor
import pdb

T = TypeVar("T")
D = TypeVar("D")

Cache = Tuple[Tensor, Tensor] | None


def default(var: T | None, val: D) -> T | D:
    return val if var is None else var


class MambaSSM(torch.nn.Module):
    def __init__(
        self, dim: int, d_state: int, h_dim: int, d_time: int, kernel_size: int
    ):
        super().__init__()
        """_summary_
        #https://arxiv.org/pdf/2312.00752
        h_t = A'*h_t-1 + B_t'x_t
        y_t = C_t*h_t
        Args:
            
            dim - dimentionality of the input data
            d_state - dimentionality of the state space in the SSM (In simplest caset d_state = h_dim)
            h_dim - dimentionality of the state space
            d_time - dimentionality of the discrete time space
            
        """

        self.linear_1 = torch.nn.Linear(dim, h_dim)
        self.linear_2 = torch.nn.Linear(dim, h_dim)
        self.conv1d = torch.nn.Conv1d(
            h_dim,
            h_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=h_dim,
            bias=True,
        )
        self.linear_f = torch.nn.Linear(h_dim, dim)

        # discretization:
        self.A = torch.nn.Parameter(
            torch.arange(1, d_state + 1, dtype=torch.float32).repeat(h_dim, 1)
        )

        # dt = torch.Parameter(dim)
        # self._A = torch.exp(self.A*dt)
        # self._B = torch.inverse(dt*self.A) * (self._A - 1) * dt*self.A
        self._B = torch.nn.Linear(h_dim, d_state)
        self._C = torch.nn.Linear(h_dim, d_state)
        self._D = torch.nn.Linear(h_dim, h_dim)

        self.D = torch.nn.Parameter(torch.ones(h_dim, dtype=torch.float32))
        self.device = torch.device("cpu")
        self.cache = True
        return

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
    ):

        # compute the discretization
        a = self.linear_1(x)
        _b = self.linear_2(x)
        prev_hid, prev_inp = default(cache, (None, None))
        b, l, d = x.size()
        x = einops.rearrange(a, "b l d -> b d l")

        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)

        a = self.conv1d(x)[..., :l]
        a = einops.rearrange(a, "b d l -> b l d")
        a = torch.nn.functional.silu(a)

        ##############################################################
        #
        #               Apply SSM:
        #
        #############################################################
        delta = torch.nn.functional.softplus(self.D + self._D(a))
        B, C = self._B(a), self._C(a)
        A_ = einops.einsum(torch.exp(-self.A), delta, "d s, b l d -> b l d s")
        B_ = einops.einsum(B, delta, " b l s, b l d -> b l d s")
        X_ = einops.einsum(B_, a, "b l d s, b l d -> b l d s")

        #####  Calculate the final state using the evolution equation #####
        b, l, d, s = A_.shape
        A_t = einops.rearrange(A_, "b l d s -> l b d s")
        X_t = einops.rearrange(X_, "b l d s -> l b d s")
        if prev_hid is not None:
            prev_hid = einops.rearrange(A_t * prev_hid + X_t, "l b d s -> b l d s")
        else:
            stack = []
            h = torch.zeros(b, d, s)
            for A_t_i, X_t_i in zip(A_t, X_t):
                h = A_t_i * h + X_t_i
                stack.append(h)
            prev_hid = torch.stack(stack, dim=1)
        ###############################################################

        output = einops.einsum(prev_hid, C, "b l d s, b l s -> b l d") + (+self.D) * a

        ###############################################################
        _b = torch.nn.functional.silu(_b)
        output = self.linear_f(a * _b)
        if self.cache:
            cache = (prev_hid.squeeze(), a[..., 1:])

        # compute the SSM
        return output, cache


if __name__ == "__main__":
    mamba = MambaSSM(dim=32, h_dim=32, d_state=32, d_time=32, kernel_size=2)
    x = torch.rand(32, 32, 32)
    cache = (None, torch.zeros(32, 32, 2 - 1))  # , device=self.device))
    result = mamba(x, cache)
