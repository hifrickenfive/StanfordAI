import utils   # The code to test

def test_pads_sents_basic():
    sents = [['hi', 'my'], ['slim'], ['ha', 'ha', 'ha', 'ha']]
    pad_token = 'butt_hole'
    output = utils.pad_sents(sents, pad_token)
    assert output == [['hi', 'my', 'butt_hole', 'butt_hole'], ['slim', 'butt_hole', 'butt_hole', 'butt_hole'], ['ha', 'ha', 'ha', 'ha']]
