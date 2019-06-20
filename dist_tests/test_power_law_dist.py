#!/usr/bin/env python

# Test power law distribution
# ---
# Abhimat Gautam

import numpy as np

import sys
sys.path.append('../')

from distributions import power_law_dist

test_periods = np.arange(1., 1000., 20.)
period_dist = power_law_dist(limits=np.array([1., 1000.]), pl_exp=-0.55)
period_p_x = period_dist.p_x(test_periods)

print(period_p_x)

test_qs = np.arange(0., 1., 0.01)
q_dist = power_law_dist(limits=np.array([0.1, 1.]), pl_exp=-0.1)
q_p_x = q_dist.p_x(test_qs)

print(q_p_x)

test_eccs = np.arange(0., 1., 0.01)
ecc_dist = power_law_dist(limits=np.array([0., 1.]), pl_exp=-0.45)
ecc_p_x = ecc_dist.p_x(test_eccs)

print(ecc_p_x)