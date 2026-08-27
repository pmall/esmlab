"""Shared pytest configuration for the esmlab test suite.

Forces the headless ``Agg`` matplotlib backend before any plotting module is
imported, so tests that render PNGs (directly or via :mod:`esmlab.plotting`)
never require a display.
"""

import matplotlib

matplotlib.use("Agg")
