import pytest

from predictor.preprocessor import NegativeSampleGenerator
from predictor.vocabulary import Vocab


@pytest.fixture
def vocab():
    return Vocab(["ATGC"])


def test_negative_sample_generator(vocab):
    gen = NegativeSampleGenerator(vocab)
    pos = 1, 2, 2, 1, 4]
    neg = gen.generate([)
    assert neg != pos
