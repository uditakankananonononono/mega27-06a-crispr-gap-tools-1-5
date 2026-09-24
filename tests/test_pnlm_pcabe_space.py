import json

def test_pnlm_pcabe_space():
    d = json.load(open('results/pnlm_pcabe_space.json'))
    pe = d['per_editor_pam']
    assert len(pe) == 6
    # LM-designed pcABE has the lowest pathogenic-bystander rate per precise edit
    for pam in ('NGG', 'NG'):
        assert pe[f'{pam}-PNLM-pcABE']['pathogenic_bystander_per_precise'] < pe[f'{pam}-ABE8e']['pathogenic_bystander_per_precise'] / 2
    # NG PAM roughly triples the precise space for ABE8e
    assert d['abe8e_precise_ng_vs_ngg_expansion'] > 2.5
    # ABE8e precision fraction under half
    assert pe['NGG-ABE8e']['precision_fraction'] < 0.4
