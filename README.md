# Research (Master & Summer Project)

*This folder contains my notebooks and codes for my summer project and master's research.*

*Last update: 2026-08-26*

___

**Research Subject**

*Stochastic Model*

- Working on classic underlying models such as G2++ (then Heston, GARCH, etc.) for training/testing/comparing deep hedging

*Deep Hedging*

- Using reinforcement learning for optimal trading policy. Implemented simple deep hedging (on Black-Scholes). Working on more complex framework such as transaction costs, different risk measures, different underlying models, etc.

*Option Surface Modeling (Fitting and Simulating)*

- Modeling arbitrage-free smooth surface at single point in time, then simulate such surface while staying smooth and arbitrage-free. Simulate with different model (simple time series, complex time series, generative AI)

<br>

**Objective**

1. Data - (*Option Surface Modeling*) -> Historical fitted option surfaces - (*Dimension Reduction*) -> Time series of option surfaces

2. Simulate option surfaces and stock price paths using stochastic models or generative AI/ML models

3. Train deep hedging agent on such paths to learn optimal trading policy for risk management, hedging, investment strategies

___

### Folders & Files

<br>

**Data**

ICAP_2Y.xlsx
- Data on caps & floors

TSLA_OptionData.csv
- Option data of TSLA (non-dividend paying stock, american options). For surface/smile calibration using SANOS.

<br>

**Deep Hedging**

BlackScholes.py
- Black-Scholes class for option pricing, greeks and Monte-Carlo simulation

Deep_hedging.ipynb
- Deep hedging theory and class for applying an agent to learn the hedging policy based on a risk measure (CVaR, MSE, SMSE) on an underlying stock path (can be any equity model), with or without transaction costs

G2++Model.ipynb
- G2++ short rate model theory and function to simulate underlying short rate paths; simulate a zero coupon bond surface; price caps & floors through analytical formula, montecarlo simulation and binomial tree; price swaptions through semi-analytical formula (Schrager & Pelsser), montecarlo and binomial tree.

Ploting_DH.py
- Python code to plot some interesting graph for the deep hedging algorithm analysis, such as ploting a sample of test paths, plot histogram of pnl, plot in a heatmap the delta (rebalancing policy) and comparing 2 methods deltas, plot deltas correlation.

Simple_Deep_hedging.ipynb
- Basic deep learning agent who can hedge Black-Scholes paths (no transaction costs)

<br>

**Notes**

Deep_learning.ipynb
- Notes on deep learning theory

Master_Research_Notes.ipynb
- Tracks of what I have done, what to read next, what to do next for master's research
- Currently focused on option surface modeling
  
Notes.ipynb
- General notes

Paper_Review.ipynb
- Overview of some research paper I read

Summer_finals.ipynb
- Overview of my summer research project to present to profs

<br>

**Other & Testing**
- General files for testing

<br>

**Vol_Models**

DYSANOS.ipynb
- Theory of DYSANOS.

Notes_VolModels.ipynb
- Notes on the volatility surface model, exploring papers such as SANOS, DYSANOS, and exploring to utilize generative model in such volatility surface model

SANOS_theory_implementation.ipynb
- Theory and implementation of SANOS to TSLA option data.
- Currently only for a single expiry, need data and adjust for full surface

sanos_---.py
- Hans Buehler SANOS code (for review, generalized code) 
