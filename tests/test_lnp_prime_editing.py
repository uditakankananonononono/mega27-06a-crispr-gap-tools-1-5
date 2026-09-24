import json

d = json.load(open('results/lnp_prime_editing.json'))


def test_dose_response_saturates_but_indels_climb():
    dr = d['dose_response']
    assert len(dr) == 4
    assert 0 < d['editing_gain_2_to_4_mgkg_pct'] < 15
    assert d['indel_gain_2_to_4_mgkg_pct'] > 40
    assert d['peak_precision_dose'] in ('1 mg/kg', '2 mg/kg')


def test_liver_confinement():
    tc = d['tissue_confinement']
    assert 40 < tc['liver'] < 50
    assert tc['max_other_tissue'] < 0.2
    assert tc['fold_liver_over_max_other'] > 100
    cc = d['celltype_confinement']
    assert cc['max_nonhepatocyte'] < 0.05


def test_aav_heart_spillover():
    a = d['aav_vs_lnp']
    assert 6 < a['aav_heart'] < 10
    assert a['lnp_heart'] < 0.05
    assert a['heart_fold_aav_over_lnp'] > 100
    assert abs(a['aav_liver'] - a['lnp_liver']) < 5


def test_offtarget_panel():
    assert len(d['offtarget_sites']) == 14
    ot13 = d['offtarget_sites']['OT13']
    assert ot13['untreated'] > 3.0
    assert ot13['PE-AAV'] > ot13['untreated']
    assert d['offtarget_max_abs_delta']['PE-AAV'] > 1.0
    # all non-OT13 sites stay within 0.2pp of background
    for s, r in d['offtarget_sites'].items():
        if s == 'OT13':
            continue
        assert abs(r['PE-AAV'] - r['untreated']) < 0.2
        assert abs(r['PE-LNP'] - r['untreated']) < 0.2


def test_motif_and_editor_rankings():
    m = d['motif_topdose_editing']
    assert m['eSBRMV1-A'] > 3 * m['tevopreQ1']
    assert m['tevopreQ1'] > m['tevo2']
    e = d['editor_topdose_editing']
    assert e['PE6d'] > e['PE6c'] > e['PEmax']
    v = d['editor_variants_ed9']
    assert v['PE6c']['editing'] > v['PE6c-La']['editing'] > v['Pemax']['editing']
