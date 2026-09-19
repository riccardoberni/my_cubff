# -*- coding: utf-8 -*-
"""
paper_constants.py
====================

Single source of truth for the numbers taken from Aguera y Arcas et al.,
"Computational Life: How Well-formed, Self-replicating Programs Emerge
from Simple Interaction" (arXiv:2406.19108), used to compare our own run
results against the paper's.

This file contains no logic, only constants - it should be imported by
any script that needs to cite one of these numbers (reseed_and_track.py,
run_metrics.py, ...) instead of retyping them by hand in several places.
If a value turns out to be wrong, it only needs to be fixed here.
"""

# Section 2.1, text after Fig. 5: soup with one replicator (Fig. 4) placed
# at a random position, run for 128 epochs, repeated over 1000 runs:
# state transition observed in 22% of cases.
PAPER_SEEDED_EPOCHS = 128
PAPER_SEEDED_TRANSITION_RATE = 0.22

# Same Section 2.1: purely randomly initialized soup (no replicator
# placed), same duration (128 epochs), 1000 runs: transition observed in
# 3 runs out of 1000.
PAPER_UNSEEDED_EPOCHS = 128
PAPER_UNSEEDED_TRANSITION_RATE = 3 / 1000  # 0.3%

# Fig. 5: 40% transition rate within 16k epochs (unseeded soup, default
# mutation rate).
PAPER_UNSEEDED_LONG_EPOCHS = 16000
PAPER_UNSEEDED_LONG_TRANSITION_RATE = 0.40

# Section 2.1, "primordial soup simulations": default soup size used in
# all of the paper's main experiments (2^17 programs).
PAPER_NUM_PROGRAMS = 131072

# Fig. 6 caption: "Distribution of the number of epochs needed for
# complexity to reach AT LEAST 1" - this is, explicitly, the operational
# definition of "state transition" used by the paper for the
# seeded/unseeded experiments above. It is NOT 2 or 3 (values used by
# analogy elsewhere in the code, but that is not the criterion the paper
# uses for THIS experiment).
PAPER_TRANSITION_ENTROPY_THRESHOLD = 1.0

# Section 2.1: default background mutation rate used in the paper
# (denominator 1<<30, numerator 1<<18 in the C++ code - see common.h).
PAPER_DEFAULT_MUTATION_RATE = (1 << 18) / (1 << 30)  # ~0.0244%