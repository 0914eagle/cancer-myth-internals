

def test_transfer_auroc_is_cross_validated():
    import numpy as np

    from src.probes import cv_auroc_transfer

    rng = np.random.default_rng(0)
    n, d = 200, 16
    y = np.array([1] * 100 + [0] * 100)
    # X_fit carries the label; X_eval is pure noise, so a leak-free transfer
    # cannot beat chance by much, while a leaky one (direction fit on all
    # items, scored on the same items) would not either -- the leak shows up
    # only when X_eval shares item-specific structure with X_fit.
    X_fit = rng.normal(size=(n, d)) + y[:, None] * 3.0
    X_eval = rng.normal(size=(n, d))
    auroc, _ = cv_auroc_transfer(X_fit, X_eval, y, [str(i) for i in range(n)])
    assert 0.35 < auroc < 0.65
    # Item-shared signal transfers.
    shared = rng.normal(size=(n, d))
    X_fit2 = shared + y[:, None] * 3.0
    X_eval2 = shared + y[:, None] * 3.0 + rng.normal(size=(n, d))
    auroc2, _ = cv_auroc_transfer(X_fit2, X_eval2, y, [str(i) for i in range(n)])
    assert auroc2 > 0.9
