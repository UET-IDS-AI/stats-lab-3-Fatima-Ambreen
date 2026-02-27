"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():

    # ---------- THEORETICAL ----------

    total_cards = 52
    aces = 4

    # P(A)
    P_A = aces / total_cards

    # P(B) (symmetry – still 4 aces out of 52)
    P_B = aces / total_cards

    # P(B | A)
    P_B_given_A = (aces - 1) / (total_cards - 1)

    # P(A ∩ B)
    P_AB = P_A * P_B_given_A

    # ---------- SIMULATION ----------

    rng = np.random.default_rng(42)
    experiments = 200000

    count_A = 0
    count_A_and_B = 0

    for _ in range(experiments):
        deck = np.array([1]*4 + [0]*48)  # 1 = Ace, 0 = not Ace
        draw = rng.choice(deck, size=2, replace=False)

        if draw[0] == 1:
            count_A += 1
            if draw[1] == 1:
                count_A_and_B += 1

    empirical_P_A = count_A / experiments

    if count_A > 0:
        empirical_P_B_given_A = count_A_and_B / count_A
    else:
        empirical_P_B_given_A = 0

    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    )


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):

    # ---------- THEORETICAL ----------
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    # ---------- SIMULATION ----------
    rng = np.random.default_rng(42)
    samples = rng.binomial(1, p, 100000)

    empirical_P_X_1 = np.mean(samples)

    absolute_error = abs(theoretical_P_X_1 - empirical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    )


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):

    # ---------- THEORETICAL ----------
    def binom_pmf(k):
        return math.comb(n, k) * (p**k) * ((1-p)**(n-k))

    theoretical_P_0 = binom_pmf(0)
    theoretical_P_2 = binom_pmf(2)
    theoretical_P_ge_1 = 1 - theoretical_P_0

    # ---------- SIMULATION ----------
    rng = np.random.default_rng(42)
    samples = rng.binomial(n, p, 100000)

    empirical_P_ge_1 = np.mean(samples >= 1)

    absolute_error = abs(theoretical_P_ge_1 - empirical_P_ge_1)

    return (
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    )


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():

    p = 1/6

    # ---------- THEORETICAL ----------
    theoretical_P_1 = p
    theoretical_P_3 = ((5/6)**2) * p
    theoretical_P_gt_4 = (5/6)**4

    # ---------- SIMULATION ----------
    rng = np.random.default_rng(42)
    samples = rng.geometric(p, 200000)

    empirical_P_gt_4 = np.mean(samples > 4)

    absolute_error = abs(theoretical_P_gt_4 - empirical_P_gt_4)

    return (
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    )


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):

    # ---------- THEORETICAL ----------
    def poisson_pmf(k):
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    theoretical_P_0 = poisson_pmf(0)
    theoretical_P_15 = poisson_pmf(15)

    # P(X ≥ 18)
    cumulative = sum(poisson_pmf(k) for k in range(18))
    theoretical_P_ge_18 = 1 - cumulative

    # ---------- SIMULATION ----------
    rng = np.random.default_rng(42)
    samples = rng.poisson(lam, 100000)

    empirical_P_ge_18 = np.mean(samples >= 18)

    absolute_error = abs(theoretical_P_ge_18 - empirical_P_ge_18)

    return (
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    )
