# Research (Master & Summer Project)

*This folder contains my notebooks and code for my summer project and master's research.*

*Last update: 2026-08-21*

### Data

ICAP_2Y.xlsx
- Data on caps & floors

TSLA_OptionData.csv
- Option data of TSLA (non-dividend paying stock, american options). For surface/smile calibration using SANOS.


### Deep Hedging

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


### Notes

Deep_learning.ipynb
- Notes on deep learning theory

Notes.ipynb
- General notes

Paper_Review.ipynb
- Overview of some research paper I read

### Other & Testing
- General files for testing

### Vol_Models

Notes_VolModels.ipynb
- Notes on the volatility surface model, exploring papers such as SANOS, DYSANOS, and exploring to utilize generative model in such volatility surface model

SANOS_theory_implementation.ipynb
- Theory and implementation of SANOS to TSLA option data.

sanos_---.py
- Hans Buehler SANOS code (for review, generalized code) 
