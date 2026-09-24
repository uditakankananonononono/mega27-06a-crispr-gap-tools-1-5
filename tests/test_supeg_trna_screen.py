import json

def test_supeg_trna_screen():
    d = json.load(open('results/supeg_trna_screen.json'))
    assert d['n_epegrnas'] > 50000
    # only a minority of epegRNAs enrich
    assert 0.1 < d['overall_frac_enriched_gt1'] < 0.25
    pc = d['per_codon']
    # codon hierarchy TAG > TGA > TAA
    assert pc['TAG']['frac_enriched'] > pc['TGA']['frac_enriched'] > pc['TAA']['frac_enriched']
    # PBS length is not the differentiator among successful designs
    fe = [v['median_fe'] for v in d['by_pbs_length_among_enriched'].values()]
    assert max(fe) / min(fe) < 1.1
    assert d['sorted_replicate_spearman'] < 0.5  # noisy low-end screen
