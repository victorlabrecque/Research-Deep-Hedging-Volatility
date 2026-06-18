"""
Deep Hedging -- Buehler, Gonon, Teichmann, Wood (2018)

Derivative  : European call option  Z = max(S1_T - K, 0)
Market      : Heston model with two hedging instruments
                S1 -- stock price
                S2 -- variance swap  (eq. 5.3-5.4 in paper)
Network     : Semi-recurrent feedforward, one sub-net per time step
                input  (log S1_k, V_k, d1_{k-1}, d2_{k-1})  ->  4 = 2d nodes
                hidden  two layers of  d+15 = 17  nodes, BatchNorm + ReLU
                output  (d1_k, d2_k)  ->  d = 2 nodes

Risk measure: MSE / variance-optimal hedging  (paper eq. 3.3 with ell(x) = x^2)

    J(theta, p0) = E[(Z - p0 - PnL)^2]

    p0 is a trainable scalar.  At the optimum:
        d/dp0 J = 0  =>  p0* = E[Z - PnL*] = E[Z] = q  (risk-neutral price)

    So p0 converges to the risk-neutral option price during training,
    while the network delta converges to the variance-minimising hedge.

Training    : Adam, lr=0.005, batch=256  (paper Section 5.1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# -- Reproducibility -----------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

# -- Heston model parameters  (paper Section 5.2) ------------------------------
S0 = 100.0  # initial stock price
V0 = 0.04  # initial variance
KAPPA = 1.0  # mean-reversion speed            (alpha in paper)
THETA = 0.04  # long-run variance               (b in paper)
XI = 2.0  # vol of vol                      (sigma in paper)
RHO = -0.7  # S-V correlation
T = 30 / 365  # 30-day horizon
N = 30  # daily rebalancing steps
K = S0  # ATM call strike

# -- Experiment / training parameters  (paper Section 5.1) --------------------
D = 2  # hedging instruments: stock + variance swap
HIDDEN = D + 15  # = 17  hidden nodes per layer
N_TRAIN = 100_000  # training paths
N_TEST = 500_000  # test paths
BATCH = 256
LR = 0.005
N_EPOCHS = 100  # ~39 000 gradient steps


# ==============================================================================
# 1.  Heston path simulation
# ==============================================================================


def simulate_heston(n_paths: int):
    """
    Euler-Maruyama discretisation of the Heston model (full-truncation scheme).

    Returns
    -------
    S1 : (n_paths, N+1)  stock price
    V  : (n_paths, N+1)  instantaneous variance
    S2 : (n_paths, N+1)  variance-swap price  (eq. 5.3-5.4)
    """
    dt = T / N

    S1 = np.zeros((n_paths, N + 1), dtype=np.float32)
    V = np.zeros((n_paths, N + 1), dtype=np.float32)
    S1[:, 0] = S0
    V[:, 0] = V0

    for k in range(N):
        Z1 = np.random.randn(n_paths).astype(np.float32)
        Z2 = np.random.randn(n_paths).astype(np.float32)
        Wv = RHO * Z1 + np.sqrt(1.0 - RHO**2) * Z2  # correlated BM for V

        Vk = np.maximum(V[:, k], 0.0)  # full truncation
        V[:, k + 1] = np.maximum(
            V[:, k] + KAPPA * (THETA - Vk) * dt + XI * np.sqrt(Vk * dt) * Wv,
            0.0,
        )
        S1[:, k + 1] = S1[:, k] * np.exp(-0.5 * Vk * dt + np.sqrt(Vk * dt) * Z1)

    # Variance swap:  S2_t = integral_0^t V_s ds  +  L(t, V_t)
    # L(t, v) = (v - theta)/kappa * (1 - exp(-kappa*(T-t))) + theta*(T-t)
    times = np.arange(N + 1, dtype=np.float32) * dt

    int_V = np.zeros_like(V)
    for k in range(1, N + 1):
        int_V[:, k] = int_V[:, k - 1] + 0.5 * (V[:, k - 1] + V[:, k]) * dt

    S2 = np.empty_like(V)
    for k in range(N + 1):
        t = times[k]
        L = (V[:, k] - THETA) / KAPPA * (1.0 - np.exp(-KAPPA * (T - t))) + THETA * (
            T - t
        )
        S2[:, k] = int_V[:, k] + L

    return S1, V, S2


def make_tensors(S1, V, S2):
    logS1 = torch.from_numpy(np.log(S1))
    V_t = torch.from_numpy(V)
    dS1 = torch.from_numpy(S1[:, 1:] - S1[:, :-1])  # (n_paths, N)
    dS2 = torch.from_numpy(S2[:, 1:] - S2[:, :-1])  # (n_paths, N)
    Z = torch.from_numpy(np.maximum(S1[:, -1] - K, 0.0))
    return logS1, V_t, dS1, dS2, Z


# ==============================================================================
# 2.  Semi-recurrent hedging network  (paper Section 4.2 and 5.1)
# ==============================================================================


class HedgingNet(nn.Module):
    """
    One independent feed-forward sub-network F_{theta_k} per rebalancing date.

    Input at step k : (log S1_k,  V_k,  delta1_{k-1},  delta2_{k-1})  -- 2d = 4
    Output          : (delta1_k,  delta2_k)                            -- d  = 2

    Architecture exactly as in paper Section 5.1:
        L=3,  N0=2d=4,  N1=N2=d+15=17,  N3=d=2
    BatchNorm before each ReLU; separate weights for every time step k.
    """

    def __init__(self):
        super().__init__()
        inp = 2 * D
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(inp, HIDDEN),
                    nn.BatchNorm1d(HIDDEN),
                    nn.ReLU(),
                    nn.Linear(HIDDEN, HIDDEN),
                    nn.BatchNorm1d(HIDDEN),
                    nn.ReLU(),
                    nn.Linear(HIDDEN, D),
                )
                for _ in range(N)
            ]
        )

    def forward(self, logS1, V, dS1, dS2):
        """
        Returns pnl (B,) = sum_k [delta1_k * dS1_k + delta2_k * dS2_k]
        """
        B = logS1.size(0)
        prev_delta = logS1.new_zeros(B, D)
        pnl = logS1.new_zeros(B)

        for k in range(N):
            inp = torch.cat([logS1[:, k : k + 1], V[:, k : k + 1], prev_delta], dim=1)
            delta = self.nets[k](inp)
            pnl += delta[:, 0] * dS1[:, k] + delta[:, 1] * dS2[:, k]
            prev_delta = delta

        return pnl


# ==============================================================================
# 3.  MSE / variance-optimal hedging loss
# ==============================================================================


class MSEHedgingLoss(nn.Module):
    """
    Variance-optimal (risk-neutral) objective  --  paper eq. (3.3) with ell(x) = x^2.

        J(theta, p0) = E[ (Z - p0 - PnL)^2 ]

    p0 is a jointly-trained scalar (the option price charged upfront).
    Taking the derivative wrt p0 and setting to zero gives:

        p0* = E[Z] - E[PnL*]  =  E[Z]  =  q   (risk-neutral price)

    because S is a martingale so E[PnL] = 0 for any admissible strategy.
    Watching p0 during training therefore shows convergence to the true price.
    """

    def __init__(self):
        super().__init__()
        self.p0 = nn.Parameter(torch.tensor(0.0))  # starts at 0, learns q

    def forward(self, pnl: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return torch.mean((Z - self.p0 - pnl) ** 2)


# ==============================================================================
# 4.  Training
# ==============================================================================


def train():
    print("Simulating training paths ...")
    S1, V, S2 = simulate_heston(N_TRAIN)
    logS1, V_t, dS1, dS2, Z = make_tensors(S1, V, S2)

    # Monte-Carlo estimate of the true risk-neutral price (for comparison)
    q_mc = float(Z.mean())
    print(
        f"Monte-Carlo risk-neutral price  q = {q_mc:.4f}  (Heston, paper reports ~1.69)"
    )

    loader = DataLoader(
        TensorDataset(logS1, V_t, dS1, dS2, Z),
        batch_size=BATCH,
        shuffle=True,
    )

    model = HedgingNet()
    loss_fn = MSEHedgingLoss()
    opt = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=LR)

    loss_history = []
    p0_history = []

    print(f"Training  epochs={N_EPOCHS}  batch={BATCH}  lr={LR}")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        loss_fn.train()
        total = 0.0
        for lS, Vb, dS1b, dS2b, Zb in loader:
            opt.zero_grad()
            pnl = model(lS, Vb, dS1b, dS2b)
            loss = loss_fn(pnl, Zb)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        p0 = float(loss_fn.p0.detach())
        loss_history.append(avg)
        p0_history.append(p0)

        if epoch % 10 == 0:
            print(
                f"  epoch {epoch:4d}/{N_EPOCHS}   MSE loss = {avg:.5f}   p0 = {p0:.4f}   q = {q_mc:.4f}"
            )

    return model, loss_fn, loss_history, p0_history, q_mc


# ==============================================================================
# 5.  Evaluation
# ==============================================================================


def evaluate(model: HedgingNet, loss_fn: MSEHedgingLoss, q_mc: float):
    print("\nSimulating test paths ...")
    S1, V, S2 = simulate_heston(N_TEST)
    logS1, V_t, dS1, dS2, Z = make_tensors(S1, V, S2)

    model.eval()
    loss_fn.eval()
    with torch.no_grad():
        pnl = model(logS1, V_t, dS1, dS2)

    hedging_error = Z - pnl  # residual after hedging (does NOT subtract p0)
    p0_star = float(loss_fn.p0.detach())
    rn_error = hedging_error - p0_star  # error relative to the learned price

    print("\n--------------------------------------")
    print(f"  MC risk-neutral price  q   = {q_mc:.4f}")
    print(f"  Learned price          p0* = {p0_star:.4f}  (should converge to q)")
    print(f"  |p0* - q|                  = {abs(p0_star - q_mc):.5f}")
    print(f"  Residual std (no hedge)    = {float((Z - q_mc).std()):.5f}")
    print(f"  Residual std (deep hedge)  = {float(rn_error.std()):.5f}")
    print("--------------------------------------")

    return rn_error.numpy(), p0_star, q_mc


# ==============================================================================
# 6.  Plots
# ==============================================================================


def plot(loss_history, p0_history, rn_error, q_mc):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # -- MSE loss curve --------------------------------------------------------
    axes[0].plot(loss_history, color="steelblue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Training loss (MSE)")
    axes[0].grid(True, alpha=0.3)

    # -- p0 convergence to q ---------------------------------------------------
    axes[1].plot(p0_history, color="darkorange", label="p0 (learned)")
    axes[1].axhline(
        q_mc, color="black", lw=1.5, linestyle="--", label=f"q = {q_mc:.3f}  (MC price)"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Price")
    axes[1].set_title("Learned price  p0  converging to  q")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # -- Residual hedging error distribution -----------------------------------
    axes[2].hist(rn_error, bins=150, density=True, alpha=0.75, color="steelblue")
    axes[2].axvline(0.0, color="black", lw=1.2, linestyle="--", label="zero")
    axes[2].axvline(
        rn_error.mean(),
        color="darkorange",
        lw=1.5,
        linestyle="--",
        label=f"mean = {rn_error.mean():.4f}",
    )
    axes[2].set_xlabel("Residual hedging error  Z - p0 - PnL")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Out-of-sample residual error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = "deep_hedging_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved -> {out}")


# ==============================================================================
# 7.  Entry point
# ==============================================================================

if __name__ == "__main__":
    model, loss_fn, loss_history, p0_history, q_mc = train()
    rn_error, p0_star, q_mc = evaluate(model, loss_fn, q_mc)
    plot(loss_history, p0_history, rn_error, q_mc)
