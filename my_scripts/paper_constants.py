# -*- coding: utf-8 -*-
"""
paper_constants.py

Settings of Aguera y Arcas et al., "Computational Life: How Well-formed,
Self-replicating Programs Emerge from Simple Interaction" (arXiv:2406.19108)
used as defaults by transition_survey.py and spatial_survey.py.
"""

# Section 2.1: number of programs of the primordial soup (2^17).
PAPER_NUM_PROGRAMS = 131072

# Threshold on the high-order entropy that defines a state transition
# (Fig. 6 counts the epochs needed for complexity to reach at least 1).
PAPER_TRANSITION_ENTROPY_THRESHOLD = 1.0

# Default background mutation rate of the engine and of the paper: 1 << 18
# out of 1 << 30 (common.h), i.e. 2^-12, reported as 0.024%.
PAPER_DEFAULT_MUTATION_RATE = (1 << 18) / (1 << 30)
