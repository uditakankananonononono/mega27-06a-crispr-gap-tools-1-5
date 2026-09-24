import json

def test_pe_lnp_invivo():
    d = json.load(open('results/pe_lnp_invivo.json'))
    dr = d['dose_response']
    assert len(dr) == 5
    # epegRNA beats HM-pegRNA at every dose, most at low dose
    assert all(x['ratio'] > 2 for x in dr)
    assert dr[-1]['ratio'] > dr[0]['ratio']
    lad = {e['editor']: e for e in d['editor_ladder']}
    assert lad['PE6c (s1)']['edit'] > 10
    assert lad['PEmax (s0.2)']['edit'] < 3
    assert lad['PE6c (s1)']['indel'] < 0.4
    k = d['kinetics']
    assert k[1]['pe6c'] > k[0]['pe6c'] > k[0]['pemax']
    assert k[-1]['pemax'] < 0.01
