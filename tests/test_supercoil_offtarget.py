import json

def test_supercoil_offtarget_sm():
    d = json.load(open('results/supercoil_offtarget_sm.json'))
    rel, sc = d['relaxed'], d['supercoiled']
    assert rel['on rel']['n'] == 697
    assert rel['ot2 rel']['n'] == 1351
    assert sc['on sc']['n'] == 960
    # supercoiling shortens median contour for on-target
    assert sc['on sc']['median'] < rel['on rel']['median'] - 3
    # and broadens the distribution
    assert sc['on sc']['iqr'] > rel['on rel']['iqr']
    assert sc['ot2 sc']['iqr'] > rel['ot2 rel']['iqr']
