#!/usr/bin/env python3
"""Run only plot_fig3_big_panel from poster_plots.py (no libpressio needed)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import unittest.mock as mock
sys.modules['libpressio'] = mock.MagicMock()

import matplotlib
matplotlib.use('Agg')

# Import as module — triggers no main() because of __name__ guard
import poster_plots as pp

print("Loading data ...", flush=True)
data, data2, data_ds, data2_ds = pp.load_data()
by_method = pp.load_csv()

print("\n[Fig 3] Running plot_fig3_big_panel ...", flush=True)
pp.plot_fig3_big_panel(data_ds, data2_ds, by_method)

print("\nDone.", flush=True)
