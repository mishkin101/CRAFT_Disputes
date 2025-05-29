import pytest
import pandas as pd
from model.data import preprocess_utterances, load_pairs
from model.config import *

# Dummy vocabulary with limited known tokens
def make_dummy_voc():
    class Voc:
        def __init__(self):
            self.word2index = {'hello': 0, 'world': 1}
    return Voc()

# Sample utterance DataFrame for two conversations
@pytest.fixture
def sample_utterances():
    data = [
        {'id': 'utt0_convo1', 'text': 'hello foo world', 'conversation_id': 'convo1'},
        {'id': 'utt1_convo1', 'text': 'bar hello',     'conversation_id': 'convo1'},
        {'id': 'utt2_convo1', 'text': 'world baz',     'conversation_id': 'convo1'},
        {'id': 'utt0_convo2', 'text': 'foo world',     'conversation_id': 'convo2'},
        {'id': 'utt1_convo2', 'text': 'hello bar',     'conversation_id': 'convo2'},
    ]
    df = pd.DataFrame(data).set_index('id')
    return df

# Sample conversation-level DataFrame with splits
@pytest.fixture
def sample_convos():
    convo_ids = ['convo1', 'convo2']
    df = pd.DataFrame({
        'split': ['train', 'val'],
        label_metadata: [1, 0]
    }, index=convo_ids)
    return df


def test_preprocess_utterances_oov_and_labels(sample_utterances):
    voc = make_dummy_voc()
    df = preprocess_utterances(sample_utterances, voc)
    # All rows preserved
    assert list(df.index) == list(sample_utterances.index)
    # tokens: known words vs UNK
    assert df.loc['utt0_convo1', 'tokens'] == ['hello', 'UNK', 'world']
    assert df.loc['utt1_convo1', 'tokens'] == ['UNK', 'hello']
    # is_attack column exists and is int zero
    assert 'is_attack' in df.columns
    assert all(df['is_attack'] == 0)


def test_load_pairs_splits_and_non_overlap(sample_utterances, sample_convos):
    voc = make_dummy_voc()
    train_pairs, val_pairs, test_pairs = load_pairs(voc, sample_utterances, sample_convos)
    # Splits: train has 1 convo, val has 1, test has none
    # train_pairs comes from convo1: 2 replies (utt1, utt2)
    assert len(train_pairs) == 3
    # val from convo2 has 1 reply (idx1 only)
    assert len(val_pairs) == 2
    assert len(test_pairs) == 0

    # Ensure no overlap in comment_ids across splits
def ids(pairs): 
    return {cid for _,_,_,cid in pairs}
    assert ids(train_pairs).isdisjoint(ids(val_pairs))
    assert ids(train_pairs).isdisjoint(ids(test_pairs))
    assert ids(val_pairs).isdisjoint(ids(test_pairs))


def test_sum_of_splits_equals_convo_count(sample_convos):
    # reconstruct splits
    splits = {
        split: set(sample_convos.index[sample_convos['split'] == split])
        for split in ('train', 'val', 'test')
    }
    total = sum(len(s) for s in splits.values())
    assert total == len(sample_convos)
    # no overlap
    assert splits['train'].isdisjoint(splits['val'])
    assert splits['train'].isdisjoint(splits['test'])
    assert splits['val'].isdisjoint(splits['test'])


def test_context_length_matches_position(sample_utterances, sample_convos):
    voc = make_dummy_voc()
    train_pairs, _, _ = load_pairs(voc, sample_utterances, sample_convos)
    # For convo1, contexts for idx1 and idx2 should have lengths 1 and 2
    for context in train_pairs:
        ctx, _, _, _ = context
        print(ctx)
        print(len(ctx))
        print(context)
    print("got to end")       
    context_lengths = [len(ctx) for ctx, _, _, _ in train_pairs]
    assert context_lengths == [1, 2, 3]
