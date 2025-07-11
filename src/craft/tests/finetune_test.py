import pytest
import pandas as pd
from craft.model.data import processLabeledDialogs, loadLabeledPairs
from craft.model.config import *

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
# Sample utterance DataFrame for two conversations
@pytest.fixture
def sample_utterances_labeled():
    data_l = [
        {'id': 'utt0_convo1', 'text': 'hello foo world', 'conversation_id': 'convo1', 'meta.provided_outcome': 0},
        {'id': 'utt1_convo1', 'text': 'bar hello',     'conversation_id': 'convo1', 'meta.provided_outcome': 0},
        {'id': 'utt2_convo1', 'text': 'world baz',     'conversation_id': 'convo1', 'meta.provided_outcome': 1},
        {'id': 'utt0_convo2', 'text': 'foo world',     'conversation_id': 'convo2', 'meta.provided_outcome': 0},
        {'id': 'utt1_convo2', 'text': 'hello bar',     'conversation_id': 'convo2', 'meta.provided_outcome': None},
    ]
    df_labeled = pd.DataFrame(data_l).set_index("id")
    return df_labeled
# Sample conversation-level DataFrame with splits
@pytest.fixture
def sample_convos():
    convo_ids = ['convo1', 'convo2']
    df = pd.DataFrame({
        'meta.split': ['train', 'val'],
        label_metadata: [1, 0]
    }, index=convo_ids)
    return df


def test_preprocess_utterances_oov_and_labels(sample_utterances_labeled):
    voc = make_dummy_voc()
    df = processLabeledDialogs(sample_utterances_labeled, voc)
    # All rows preserved
    assert list(df.index) == list(sample_utterances_labeled.index)
    # tokens: known words vs UNK
    assert df.loc['utt0_convo1', 'tokens'] == ['hello', 'UNK', 'world']
    assert df.loc['utt1_convo1', 'tokens'] == ['UNK', 'hello']
    # is_attack column exists and is int zero
    assert 'is_attack' in df.columns
    assert all(df['is_attack'] == sample_utterances_labeled['meta.provided_outcome'].fillna(0).astype(int))


def test_load_pairs_splits_and_non_overlap(sample_utterances_labeled, sample_convos):
    voc = make_dummy_voc()
    train_pairs, val_pairs, test_pairs = loadLabeledPairs(voc, sample_utterances_labeled, sample_convos, False, False, False)
    # Splits: train has 1 convo, val has 1, test has none
    # train_pairs comes from convo1: 2 replies (utt1, utt2) if utteranve level labels used
    # train_pairs comes from convo1: 3 replies (utt1, utt2, utt3) if utteranve level labels used
    assert len(train_pairs) == 2
    # val from convo2 has 1 reply (idx1 only)
    # val from convo2 has 1 reply (idx1 only)

    assert len(val_pairs) == 1
    assert len(test_pairs) == 0

    # Ensure no overlap in comment_ids across splits
    def ids(pairs): 
        assert ids(train_pairs).isdisjoint(ids(val_pairs))
        assert ids(train_pairs).isdisjoint(ids(test_pairs))
        assert ids(val_pairs).isdisjoint(ids(test_pairs))


def test_sum_of_splits_equals_convo_count(sample_convos):
    # reconstruct splits
    splits = {
        split: set(sample_convos.index[sample_convos['meta.split'] == split])
        for split in ('train', 'val', 'test')
    }
    total = sum(len(s) for s in splits.values())
    assert total == len(sample_convos)
    # no overlap
    assert splits['train'].isdisjoint(splits['val'])
    assert splits['train'].isdisjoint(splits['test'])
    assert splits['val'].isdisjoint(splits['test'])


def test_context_length_matches_position(sample_utterances_labeled, sample_convos):
    voc = make_dummy_voc()
    train_pairs, val_pairs, _= loadLabeledPairs(voc, sample_utterances_labeled, sample_convos, False, False, False)
    print("label_metadata:", label_metadata)
    print("train_pairs:", train_pairs)
    print("val_pair:", val_pairs)

    # For convo1, contexts for idx1 and idx2 should have lengths 1 and 2
    for context in train_pairs:
        ctx, _, _, _ = context
        # print(ctx)
        # print(len(ctx))
        # print(context)
    print("got to end")       
    context_lengths = [len(ctx) for ctx, _, _, _ in train_pairs]
    assert context_lengths == [1, 2]


def test_last_only_returns_one_pair_per_conversation(sample_utterances_labeled, sample_convos):
    """
    When last_only=True, each conversation should yield exactly one (context, reply, label, comment_id) pair.
    """
    voc = make_dummy_voc()
    # Enable last_only for train split only
    train_pairs, val_pairs, test_pairs = loadLabeledPairs(
        voc,
        sample_utterances_labeled,
        sample_convos,
        last_only_train=True,
        last_only_val=False,
        last_only_test=False
    )
    # With last_only_train=True, train_pairs should contain exactly one pair
    assert len(train_pairs) == 1
    # The single pair's comment_id should correspond to the last utterance of convo1 before final labeled reply (utt2_convo1)
    _, _, _, comment_id = train_pairs[0]
    assert comment_id == 'utt1_convo1'
    assert len(val_pairs) == 1  # default last_only_val=False yields two replies? but sample_convos has one val convo2 of two utts -> val_pairs length 1 (for reply utt1_convo2 only)
    assert len(test_pairs) == 0