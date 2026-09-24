import torch

from crisprgap.models.efficacy_cnn import GuideEfficacyCNN, make_aux_features
from crisprgap.models.offtarget_gnn import (OffTargetGNN, duplex_to_graph,
                                            collate_graphs, NODE_DIM)


def test_efficacy_cnn_forward_shapes():
    model = GuideEfficacyCNN(seq_len=30)
    x = torch.randn(8, 4, 30)
    aux = torch.randn(8, 17)
    out = model(x, aux)
    assert out.shape == (8,)


def test_efficacy_aux_feature_dim():
    f = make_aux_features("ACGT" * 8)
    assert f.shape == (17,)


def test_efficacy_cnn_gradient_flows():
    model = GuideEfficacyCNN()
    x = torch.randn(4, 4, 30)
    aux = torch.randn(4, 17)
    loss = model(x, aux).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_duplex_graph_perfect_match():
    x, ei = duplex_to_graph("ACGT", "ACGT")
    assert x.shape == (4, NODE_DIM)
    assert x[:, 16].sum() == 0.0
    assert ei.shape[0] == 2


def test_duplex_graph_mismatch_edges():
    x, ei = duplex_to_graph("AAAA", "ATTA")
    assert x[:, 16].sum() == 2.0
    mm_pairs = [(int(a), int(b)) for a, b in ei.t().tolist() if a in (1, 2) and b in (1, 2)]
    assert (1, 2) in mm_pairs and (2, 1) in mm_pairs


def test_offtarget_gnn_single_graph_scalar():
    model = OffTargetGNN()
    x, ei = duplex_to_graph("ACGTACGTACGTACGTACGT", "ACGTTCGTACGTACGTAGGT")
    out = model(x, ei)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_offtarget_gnn_batched_matches_single():
    model = OffTargetGNN()
    pairs = [("ACGT" * 5, "ACGT" * 5), ("A" * 20, "T" * 20)]
    graphs = [duplex_to_graph(g, o) for g, o in pairs]
    xb, eib, batch = collate_graphs(graphs)
    out_batch = model(xb, eib, batch)
    assert out_batch.shape == (2,)
    singles = [model(x, ei) for x, ei in graphs]
    for b, s in zip(out_batch, singles):
        assert torch.allclose(b, s[0], atol=1e-5)
