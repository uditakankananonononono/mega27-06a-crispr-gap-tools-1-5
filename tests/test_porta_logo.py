import os


def test_logo_figure_exists_and_is_rich():
    import numpy as np
    from PIL import Image
    p = "papers/figs/fig_porta_logo.png"
    assert os.path.exists(p)
    im = np.asarray(Image.open(p).convert("L"))
    assert (im < 250).mean() > 0.1  # actually rendered content
