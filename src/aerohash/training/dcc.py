"""Discrete cyclic coordinate updates and full ADSH objective."""
import torch

def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """Minimize the ADSH objective one binary column at a time.

    A zero coordinate target is a tie: retain the previous sign, using +1
    for an uninitialized zero. Returning 0 would violate the binary domain.
    """
    optim_dim = code_length
    Q = (optim_dim * S).t() @ U + gamma * expand_U

    for bit in range(optim_dim):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        target = q.t() - B_prime @ U_prime.t() @ u.t()
        previous_sign = torch.where(B[:, bit] < 0, -1., 1.)
        B[:, bit] = torch.where(target == 0, previous_sign, target.sign())
    return B

def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()
