import json

def test_sacas9_variants():
    d = json.load(open('results/sacas9_variants.json'))
    nuc = d['nuclease_indel']
    assert nuc['SaCas9-WT']['n_targets'] == 35
    assert nuc['SaCas9-NNG']['n_targets'] == 35
    assert nuc['SaCas9-NNG']['active_frac_gt10pct'] > nuc['SaCas9-WT']['active_frac_gt10pct']
    be = d['base_editing']
    assert be['SaCas9-NNG-AID']['active_frac_gt10pct'] > be['SaCas9-WT-AID']['active_frac_gt10pct']
    p = d['offtarget_panel']
    assert p['n_sites'] == 74
    assert 0.3 < p['eWT_read_reduction_frac'] < 0.5
    assert 0.45 < p['eNNG_read_reduction_frac'] < 0.7
    assert p['WT']['total_reads'] == 78725
    assert p['eWT']['sites_with_ge10_reads'] == 36
    assert 0.6 < p['wt_vs_ewt_spearman'] < 0.9
