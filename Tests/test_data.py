import os
import json
import tempfile
import pytest

import data
from data import Voc

class DummyUtterance:
    def __init__(self, text):
        self.text = text

class DummyConversation:
    def __init__(self, dialogs):
        self._dialogs = dialogs
    def get_root_to_leaf_paths(self):
        return self._dialogs

class DummyCorpus:
    def __init__(self, convos):
        self._convos = convos
    def iter_conversations(self):
        return iter(self._convos)

@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    # Run tests in isolated temp directory
    monkeypatch.chdir(tmp_path)
    return tmp_path

def test_voc_add_and_trim():
    voc = Voc("test")
    # initial tokens: PAD, SOS, EOS, UNK
    assert voc.num_words == 4
    voc.addSentence("hello world world")
    # new tokens hello, world
    assert "hello" in voc.word2index
    assert "world" in voc.word2index
    # world count increments
    assert voc.word2count["world"] == 2
    assert voc.num_words == 6

    # Trim words with count < 2 (removes 'hello')
    voc.trim(min_count=2)
    assert "hello" not in voc.word2index
    assert "world" in voc.word2index
    # After trimming: 4 default + 1 kept word
    assert voc.num_words == 5

def test_create_train_file(monkeypatch, isolate_cwd):
    # Prepare dummy conversations
    dialogs1 = [[DummyUtterance("a"), DummyUtterance("b")]]
    dialogs2 = [[DummyUtterance("c")]]
    convos = [DummyConversation(dialogs1), DummyConversation(dialogs2)]
    dummy_corpus = DummyCorpus(convos)

    # Monkey-patch corpus and corpus_name
    monkeypatch.setattr(data, "corpus", dummy_corpus)
    monkeypatch.setattr(data, "corpus_name", "mycorp")

    # Run function
    data.createTrainFile()

    # Check output file exists
    out_dir = os.path.join("nn_input_data", "mycorp")
    out_file = os.path.join(out_dir, "train_processed_dialogs.txt")
    assert os.path.isdir(out_dir)
    assert os.path.isfile(out_file)

    # Read lines and verify JSON content
    lines = open(out_file).read().splitlines()
    assert len(lines) == 2

    data1 = json.loads(lines[0])
    assert data1 == [{"text": "a"}, {"text": "b"}]
    data2 = json.loads(lines[1])
    assert data2 == [{"text": "c"}]
