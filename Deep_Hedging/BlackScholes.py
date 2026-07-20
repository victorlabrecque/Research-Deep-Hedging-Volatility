import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq  # solve implied vol


class BlackScholes:
    """
    Black-Scholes model
    1. option price
    2. Greeks
    3. Implied volatility
    4. Montecarlo paths simulation
    5. Deltas hedging (with and without transaction costs)
    """

    def __init__(self, S, K, r, sigma, T):
        self.S = S  # current asset price
        self.K = K  # strike
        self.r = r  # risk free rate
        self.sigma = sigma  # volatility
        self.T = T  # time to maturity

    # --- 1. Option Price ---

    def call_price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * (self.sigma**2)) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return float(
            self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )

    def put_price(self):  # by put call parity
        return float(self.call_price() - self.S + self.K * np.exp(-self.r * self.T))

    # --- 2. Greeks ---

    def call_delta(self):  # first derivative with respect to price S
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * (self.sigma**2)) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return float(norm.cdf(d1))

    def put_delta(self):  # by derivative of put-call parity with respect to S
        return self.call_delta() - 1

    def gamma(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * (self.sigma**2)) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return float(norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T)))

    # --- 3. Implied Volatility ---

    def implied_vol(self, price, call=True):
        def call_price(sigma_implied):
            d1 = (
                np.log(self.S / self.K) + (self.r + 0.5 * (sigma_implied**2)) * self.T
            ) / (sigma_implied * np.sqrt(self.T))
            d2 = d1 - sigma_implied * np.sqrt(self.T)
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(
                d2
            )

        if call:
            objective = lambda sigma_implied: call_price(sigma_implied) - price  # noqa: E731
        if not call:

            def put_price(sigma_implied):  # by put-call parity
                return (
                    call_price(sigma_implied)
                    - self.S
                    + self.K * np.exp(-self.r * self.T)
                )

            objective = lambda sigma_implied: put_price(sigma_implied) - price  # noqa: E731

        return brentq(objective, 1e-6, 10)

    # --- 4. Montecarlo simulation (exact discretization) ---

    def BS_paths(self, N_steps, M_paths, seed=123):
        np.random.seed(seed)
        dt = self.T / N_steps
        Z = np.random.randn(M_paths, N_steps)
        log_return = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        log_S = np.log(self.S) + np.cumsum(log_return, axis=1)
        S_paths = np.exp(log_S)
        S_paths = np.column_stack([np.full(M_paths, self.S), S_paths])

        return S_paths.astype(np.float32)

    # --- 5. Delta hedging (given simulated paths) ---

    def delta_hedging(
        self,
        S_paths,
        Initial_portfolio=0,
        transaction_cost=False,
        kappa=0,
        risk_aversion=1,
    ):
        # transaction_cost = 'Leland', or 'Whalley-Wilmott'

        # parameters
        ttm = np.linspace(self.T, 0, S_paths.shape[1])
        dt = self.T / (S_paths.shape[1] - 1)

        # call price value (initial value of the hedging portfolio)
        call_price_maturity = np.full(S_paths.shape[0], Initial_portfolio) * np.exp(
            self.r * self.T
        )

        # payoff (at maturity)
        payoff = np.maximum(S_paths[:, -1] - self.K, 0)

        # delta (not at maturity)
        S_grid = S_paths[:, :-1]
        T_grid = ttm[:-1]

        sigma = self.sigma
        if transaction_cost == "Leland":
            sigma = self.sigma * np.sqrt(
                1
                + np.sqrt(2 / np.pi)
                * (2 * kappa / (self.sigma * np.sqrt(risk_aversion)))
            )

        d1 = (np.log(S_grid / self.K) + (self.r + 0.5 * (sigma**2)) * T_grid) / (
            sigma * np.sqrt(T_grid)
        )
        deltas = norm.cdf(d1)

        if transaction_cost == "Whalley-Wilmott":
            gamma = norm.pdf(d1) / (S_grid * sigma * np.sqrt(T_grid))
            H = (
                (1.5 * kappa / risk_aversion)
                * np.exp(-self.r * T_grid)
                * S_grid
                * (gamma**2)
            ) ** (1 / 3)

            N_paths, N_steps = deltas.shape
            compound = np.exp(self.r * (self.T - ttm[1:]))

            # Sequential: track actual position under no-transaction band rule
            actual_deltas = np.zeros_like(deltas)
            delta_current = np.zeros(N_paths)  # start with no position

            for t in range(N_steps):
                lower = deltas[:, t] - H[:, t]
                upper = deltas[:, t] + H[:, t]

                new_delta = delta_current.copy()
                new_delta[delta_current < lower] = lower[
                    delta_current < lower
                ]  # buy to lower band
                new_delta[delta_current > upper] = upper[
                    delta_current > upper
                ]  # sell to upper band
                # else: stay — no trade

                actual_deltas[:, t] = new_delta
                delta_current = new_delta

            # PnL
            dS = S_paths[:, 1:] - S_paths[:, :-1] * np.exp(self.r * dt)
            hedging_pnl = np.sum(actual_deltas * dS * compound, axis=1)

            # Transaction costs: κ * |Δδ| * S_t, compounded to maturity
            prev_deltas = np.concatenate(
                [np.zeros((N_paths, 1)), actual_deltas[:, :-1]], axis=1
            )
            tc = kappa * np.sum(
                np.abs(actual_deltas - prev_deltas) * S_grid * compound, axis=1
            )

            pnl = call_price_maturity + hedging_pnl - payoff - tc
            print(
                f"Black-Scholes (With Transaction Costs) | mean P&L={pnl.mean():.4f} | std={pnl.std():.4f} | mean TC={tc.mean():.4f}"
            )
            return pnl, actual_deltas, tc

        else:  # pnl at each step t = 1 to T-1
            compound = np.exp(
                self.r * (self.T - ttm[1:])
            )  # compounding each gain to maturity
            dS = S_paths[:, 1:] - S_paths[:, :-1] * np.exp(
                self.r * dt
            )  # discounted stock move
            hedging_pnl = np.sum(deltas * dS * compound, axis=1)

            # total pnl
            pnl = call_price_maturity + hedging_pnl - payoff
            print(
                f"Black-Scholes (No Transaction Costs) | mean P&L={pnl.mean():.4f} | std={pnl.std():.4f}"
            )
            return pnl, deltas
