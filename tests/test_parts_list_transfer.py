import json

def test_parts_list_transfer():
    d = json.load(open('results/parts_list_transfer.json'))
    assert d['n_promoters_celltype_panel'] >= 200
    assert d['n_scaffold_variants_locus_panel'] >= 250
    # promoters transfer across cell types
    assert d['promoter_cross_celltype_min'] > 0.85
    # but not across loci
    assert d['promoter_cross_locus_min'] < 0.4
    assert d['scaffold_cross_locus_min'] < 0.4
    mx = max(d['promoter_cross_locus_spearman'].values())
    assert mx > 0.7  # some locus pairs do transfer - spread is the point
