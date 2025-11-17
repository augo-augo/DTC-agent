# Testing Guide

Run the test suite with `pytest` from the repository root (make sure the project is
available on the Python path, e.g. `pip install -e .` if you are on a fresh VM):

```bash
pytest
```

Key checks include:
- Entropy estimation sanity checks (`tests/test_motivation.py`)
- Rollout buffer behaviour and training loop wiring (`tests/test_training.py`)
- Dreaming loss regression (`tests/test_dreaming.py` – asserts numeric stability of gradients and total loss)
- Component-level coverage of GAE/normalisation utilities (`tests/test_components.py`)

Add additional scenarios mirroring new modules as they land (e.g., empowerment stability,
workspace heuristics, and optimizer behaviours). When running at scale, keep an eye on the
regression assertions so you notice behaviour changes early.
