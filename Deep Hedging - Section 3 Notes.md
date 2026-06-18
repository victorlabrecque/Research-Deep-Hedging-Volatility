# Section 3: Pricing and Hedging Using Convex Risk Measures — Detailed Walkthrough

This section builds the **theoretical optimization framework** that Section 4 later approximates with neural networks. The goal: define what an "optimal hedge" and a "fair price" mean once you allow market frictions (transaction costs, constraints) and an incomplete market.

---

## 3.0 Motivation: why we need this at all

In a frictionless, complete market, any liability $Z$ has a **unique replicating strategy** $\delta$ and a fair price $p_0$ such that

$$
-Z + p_0 + (\delta\cdot S)_T - C_T(\delta) = 0 \quad \mathbb{P}\text{-a.s.}
$$

i.e. you can perfectly cancel out the liability with no leftover risk. With frictions (transaction costs $C_T$) or trading constraints ($\delta \in \mathcal{H}$ rather than unconstrained), **perfect replication generally isn't possible**. So the agent must instead choose:

- An **optimality criterion** — a functional that scores how "good" a terminal P&L distribution is.
- A **minimal price** — the smallest amount of cash that needs to be injected so that, after hedging optimally, the position becomes "acceptable" under that criterion.

This is exactly the role of a **convex risk measure**.

---

## 3.1 Definition of a convex risk measure (Definition 3.1)

A convex risk measure $\rho: \mathcal{X} \to \mathbb{R}$ ($\mathcal{X}$ = real-valued random variables on $\Omega$, representing *asset positions*, so $-X$ is a liability) must satisfy:

1. **Monotone decreasing**:
$$X_1 \ge X_2 \implies \rho(X_1) \le \rho(X_2).$$
   *Interpretation*: a better (higher-valued) position needs less extra cash to be made acceptable.

2. **Convexity**:
$$\rho(\alpha X_1 + (1-\alpha) X_2) \le \alpha \rho(X_1) + (1-\alpha)\rho(X_2), \quad \alpha \in [0,1].$$
   *Interpretation*: diversification cannot increase risk — mixing two positions is never riskier than the weighted average of their individual risks.

3. **Cash invariance**:
$$\rho(X + c) = \rho(X) - c, \quad c \in \mathbb{R}.$$
   *Interpretation*: adding cash $c$ to a position reduces the capital you need to inject by exactly $c$. This gives $\rho$ a literal monetary meaning: $\rho(X)$ is **the minimal amount of cash you must add to $X$ to make it "acceptable"** (i.e. $\rho(X + \rho(X)) = 0$).

$\rho$ is *normalized* if $\rho(0) = 0$.

**Key example used throughout the paper**: the entropic risk measure
$$\rho(X) = \frac{1}{\lambda}\log \mathbb{E}[\exp(-\lambda X)]$$
satisfies all three axioms and ties to exponential utility (see §3.5 below).

---

## 3.2 The optimal hedging/pricing functional $\pi$ (eq. 3.1)

Given a risk measure $\rho$ and a target position $X$ (think $X = -Z$, a liability), define

$$
\pi(X) := \inf_{\delta \in \mathcal{H}} \rho\big(X + (\delta\cdot S)_T - C_T(\delta)\big).
$$

This says: **search over all admissible trading strategies $\delta$**, apply each one's resulting P&L on top of $X$, score it with $\rho$, and take the best (smallest) score. $\pi(X)$ is the **least amount of risk-measure-cash you need after hedging optimally**.

### Proposition 3.2: $\pi$ inherits the risk-measure structure

- $\pi$ is always monotone decreasing and cash-invariant (these come directly from $\rho$'s properties, since adding cash to $X$ just adds cash to the argument of $\rho$ for every $\delta$).
- If additionally $C_T(\cdot)$ and $\mathcal{H}$ are **convex** (true for proportional/convex cost structures and convex constraint sets), then $\pi$ itself is a convex risk measure.

**Proof sketch** (the convexity part is the interesting one): take $X_1, X_2$ and $\alpha \in [0,1]$. For *any* pair of strategies $\delta_1, \delta_2 \in \mathcal{H}$, the mixed strategy $\alpha\delta_1+(1-\alpha)\delta_2$ is also in $\mathcal{H}$ (convexity of $\mathcal{H}$), and by linearity of $(\delta\cdot S)_T$ in $\delta$:

$$
\alpha(\delta_1\cdot S)_T + (1-\alpha)(\delta_2\cdot S)_T = \big((\alpha\delta_1+(1-\alpha)\delta_2)\cdot S\big)_T.
$$

Combined with convexity of $C_T$ (so $C_T(\alpha\delta_1+(1-\alpha)\delta_2) \le \alpha C_T(\delta_1)+(1-\alpha)C_T(\delta_2)$) and monotonicity + convexity of $\rho$, a chain of inequalities collapses the infimum over the *mixed* strategy into the sum of the two individual infima — exactly the convexity inequality required of $\pi$. Cash-invariance/monotonicity of $\pi$ follow trivially from the corresponding property of $\rho$ (the $\inf_\delta$ doesn't interact with them).

**Why this matters**: the *pricing operator built from hedging* behaves just like a primitive risk measure — so all risk-measure theory carries over to "price of a portfolio after optimal hedging," not just to a single static position.

---

## 3.3 The indifference price (eq. 3.2)

Define the **optimal hedging strategy** as any minimizer $\delta \in \mathcal{H}$ of (3.1). Naively, you might define the "fair price" of $Z$ as $\pi(-Z)$ — the minimal capital injection making $-Z$ (plus optimal hedge) acceptable. But this has a flaw: it would imply that having *no* liabilities at all could still require or release cash — e.g. if some hedging instrument has positive expected return under $\mathbb{P}$ (the model deliberately allows $S$ to be a **non-martingale**, to embed trading signals/views). So $\pi(0)$ need not be zero.

To fix this, define the **indifference price**:

$$
p(Z) := \pi(-Z) - \pi(0).
$$

This is the cash amount $p_0$ that makes the agent **indifferent** between selling $-Z+p_0$ and hedging it, versus doing nothing ($\pi(-Z+p_0) = \pi(0)$, and by cash invariance this pins down $p_0 = p(Z)$).

### Lemma 3.3: consistency with classical replication pricing

If there are **no transaction costs** ($C_T \equiv 0$), **no constraints** ($\mathcal{H} = \mathcal{H}^u$), and $Z$ is *attainable* ($Z = p_0 + (\delta^*\cdot S)_T$ for some strategy $\delta^*$), then $p(Z) = p_0$ — the indifference price **reduces exactly to the classical replication price**. A sanity check that the new machinery doesn't break the classical complete-market answer.

*Proof idea*: For any $\delta$, cash-invariance of $\rho$ gives
$$\rho(-Z + (\delta\cdot S)_T) = p_0 + \rho\big(([\delta-\delta^*]\cdot S)_T\big).$$
Taking $\inf_\delta$ and shifting the variable $\delta \to \delta-\delta^*$ (using $\mathcal{H} - \delta^* = \mathcal{H}$ since unconstrained) gives $\pi(-Z) = p_0 + \pi(0)$, hence $p(Z) = p_0$.

---

## 3.4 Arbitrage and "irrelevance"

The paper explicitly does **not** assume the market is arbitrage-free (unusual for classical pricing theory, but natural here since $S$ may not be a martingale under $\mathbb{P}$ — $\mathbb{P}$ could be a real-world measure incorporating views).

- An **arbitrage opportunity given $X$** is a strategy $\delta^{[X]}$ such that the resulting P&L is a.s. $\ge 0$ and strictly positive with positive probability.
- If such an opportunity exists and can be scaled up without limit, $\pi(X) = -\infty$. If this happens at $X=0$, the market is called **irrelevant**.

**Corollary 3.5**: if $\pi(0) > -\infty$ then $\pi(X) > -\infty$ for *all* $X$ (since $\Omega$ is finite, $X$ is bounded, so monotonicity bounds $\pi(X)$ below by $\pi(\sup X) \ge \pi(0) - \sup X$).

Two subtleties illustrated by example:

- **Statistical arbitrage without classical arbitrage** can still make a market irrelevant: e.g. under $\rho(X) = -\mathbb{E}[X]$ with positive drift $\mu$ and cheap-enough proportional cost, you can scale a directional bet indefinitely and $\pi(0) = -\infty$. Realistic analogy: a desk systematically selling options earns money on average but with occasional large losses — this is "infinitely good" under a risk measure that doesn't penalize tail risk enough (plain expectation).
- Conversely, even with **classical arbitrage present**, a sufficiently conservative risk measure (e.g. worst-case $\rho(X) = -\inf X$) combined with a non-zero chance of zero payoff can keep the market "relevant" ($\pi(0)=0$) — the risk measure refuses to credit a strategy unless it works in *every* scenario.

This discussion is a caution: $\rho$, $\mathcal{H}$, and $C_T$ must be jointly well-posed (finite $\pi(0)$), or the later neural-network training problem is degenerate (unbounded below).

---

## 3.5 Subsection 3.1 — Exponential Utility Indifference Pricing (Lemma 3.6)

Connects the abstract $\rho$-framework to the **classical exponential-utility indifference price** $q(Z)$, defined via:

$$
\sup_{\delta\in\mathcal{H}} \mathbb{E}\big[U(q(Z) - Z + (\delta\cdot S)_T + C_T(\delta))\big]
=
\sup_{\delta\in\mathcal{H}} \mathbb{E}\big[U((\delta\cdot S)_T + C_T(\delta))\big],
$$

where $U(x) = -\exp(-\lambda x)$ (CARA utility, risk-aversion $\lambda$).

**Lemma 3.6**: if $\rho$ is the **entropic risk measure**

$$
\rho(X) = \frac{1}{\lambda}\log \mathbb{E}[\exp(-\lambda X)],
$$

then $p(Z) = q(Z)$ — the abstract indifference price (3.2) **coincides exactly** with the classical exponential-utility indifference price. This bridges the paper's numerical examples (Section 5.3, exponential utility indifference pricing under transaction costs) to the classical utility-theory language used in the transaction-cost-asymptotics literature (Whalley–Wilmott, Kallsen–Muhle-Karbe).

*Proof*: simple algebra — since $U$ is exponential, $\sup_\delta \mathbb{E}[U(\cdot)]$ factors into $-\exp(-\lambda \cdot \text{stuff})$, and taking logs turns the utility-indifference equation directly into the cash-invariance equation defining $p(Z)$.

---

## 3.6 Subsection 3.2 — Optimized Certainty Equivalents (OCE)

A **whole family** of convex risk measures, generalizing the entropic one, built from a **loss function** $\ell:\mathbb{R}\to\mathbb{R}$ (continuous, non-decreasing, convex):

$$
\rho(X) := \inf_{w\in\mathbb{R}} \Big\{ w + \mathbb{E}[\ell(-X-w)] \Big\}.
$$

**Lemma 3.7** proves this is a convex risk measure:

- *Monotonicity*: $\ell$ non-decreasing $\implies$ $X\le Y$ gives $\mathbb{E}[\ell(-X-w)] \ge \mathbb{E}[\ell(-Y-w)]$ for any $w$, so $\rho(X)\ge\rho(Y)$.
- *Cash invariance*: substituting $w \to w+m$ shows $\rho(X+m) = -m + \rho(X)$.
- *Convexity*: splitting the infimum over a joint $(w_1,w_2)$ and using convexity of $\ell$.

This construction matters practically: it turns the *risk measure* optimization into a **joint optimization over the hedging strategy $\delta$ AND a scalar $w$** — exactly how Section 4.3 sets up the neural-network training objective ($w$ becomes an extra trainable scalar alongside the network weights).

**Example 3.8 (recovering entropic risk)**: with
$$\ell(x) = \exp(\lambda x) - \frac{1+\log\lambda}{\lambda},$$
solving the inner optimization over $w$ explicitly gives back exactly the entropic risk measure (3.4). So entropic risk is a *special case* of OCE.

**Example 3.9 (CVaR / Expected Shortfall)**: with
$$\ell(x) = \frac{1}{1-\alpha}\max(x,0),$$
the OCE construction yields **Average Value at Risk** (Conditional VaR / Expected Shortfall) at level $1-\alpha$ — the risk measure actually used in the Section 5.2 numerical experiments ($\alpha = 0.5, 0.95, 0.99$ controlling risk-aversion).

---

## 3.7 Proposition 3.10 — Connecting to the martingale/risk-neutral case

Justifies the benchmark comparisons in Section 5. Assume:

- $S$ is a $\mathbb{P}$-martingale (no statistical edge baked into the price process — the "Heston under the risk-neutral measure" setting used numerically),
- $\rho$ is an OCE risk measure (3.5).

Then:

**(i)** $\pi(0) = \rho(0)$ — with no liability, the optimal-hedging price collapses to the risk measure evaluated at zero (hedging can't create value if $S$ has no drift and costs are nonnegative).

**(ii)** $p(Z) \ge \mathbb{E}[Z]$ for any $Z$ — the indifference price is **never less than the risk-neutral expectation**. Frictions/risk-aversion can only make the price *more conservative* (higher, from the seller's side) than the frictionless risk-neutral price, never less.

*Proof idea*: $\pi(0) \le \rho(0)$ trivially (take $\delta=0$, $C_T(0)=0$). For the reverse, use that $S$ being a martingale gives $\mathbb{E}[(\delta\cdot S)_T] = 0$ for all admissible $\delta$ (eq. 3.6, a discrete-time martingale-transform argument), then apply **Jensen's inequality** ($\ell$ convex) to pull the expectation inside $\ell$, plus monotonicity of $\ell$ and nonnegativity of $C_T$ to drop the cost term:

$$
\pi(-Z) = \inf_{w}\inf_{\delta} \Big\{ w + \mathbb{E}[\ell(Z-(\delta\cdot S)_T + C_T(\delta) - w)] \Big\}
\ge \inf_w \Big\{ w + \ell(\mathbb{E}[Z]-w) \Big\} = \mathbb{E}[Z] + \rho(0).
$$

Setting $Z=0$ gives $\pi(0)\ge\rho(0)$, hence (i); combining with (i) gives (ii).

This is exactly why, in the Section 5.2 Heston-model numerics with **zero transaction costs**, the paper can sanity-check the neural network against the classical risk-neutral price $q = \mathbb{E}^{\mathbb{Q}}[Z]$: the proposition guarantees the risk-adjusted price $p_0^\theta$ found by the network must sit at or above $q$ — confirmed numerically ($q = 1.69$ vs. $p_0^\theta = 1.94$ at $\alpha=0.5$, rising to $3.49$ at $\alpha=0.99$).

---

## How Section 3 sets up Section 4

Everything here is **abstract / infinite-dimensional**: $\inf_{\delta\in\mathcal{H}}$ over an enormous (functional) space of strategies. Section 4's job is to show that replacing $\mathcal{H}$ with a much smaller, finite-dimensional set $\mathcal{H}_M$ of **neural-network-parametrized strategies** still converges to the same optimal value $\pi(X)$ as network capacity $M\to\infty$ (Proposition 4.9), and that for OCE/entropic risk measures specifically, the resulting finite-dimensional problem $\inf_\theta J(\theta)$ (eq. 4.6/4.8) is *exactly* in the form

$$
\inf_{w,\theta} \Big\{ w + \mathbb{E}[\ell(\cdots)] \Big\}
$$

that Section 3.2's OCE construction produces — which is precisely what makes it trainable by stochastic gradient descent / backpropagation.