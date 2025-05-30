import os
import json
import pytest
import regex
import shutil
import emoji
from pathlib import Path

import model.data as data
from model.data import tokenize, Voc, processDialog, createTrainFile
from convokit import Corpus

import model.config as cfg
from runners.build_vocabulary_objects import build_Vocab
from runners import build_vocabulary_objects


class DummyUtterance:
    def __init__(self, text):
        self.text = text

class DummyConversation:
    def __init__(self, dialogs):
        self._dialogs = dialogs
    def get_root_to_leaf_paths(self):
        return self._dialogs

class DummyCKCorpus:
    def __init__(self, filename):
        pass
    def iter_conversations(self):
        return iter(self._convos)
    @classmethod
    def with_convos(cls, convos):
        obj = cls(None)
        obj._convos = convos
        return obj


@pytest.fixture
def isolate_cwd(tmp_path, monkeypatch):
    """
    Change into a temp dir and point the module-level
    repo_dir, corpus_name, corpus_dir at a fake tree.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "repo_dir",    str(tmp_path),   raising=False)
    monkeypatch.setattr(data, "corpus_name", "mycorp",       raising=False)
    fake_root = tmp_path / "fake_corpora"
    (fake_root / "one").mkdir(parents=True)
    (fake_root / "two").mkdir()
    out_dir = tmp_path / "data" / "mycorp"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(data, "corpus_dir", str(fake_root),   raising=False)
    monkeypatch.setattr(cfg, "word2index_path",str(out_dir / "word2index.json"),raising=False)
    monkeypatch.setattr(cfg, "index2word_path",str(out_dir / "index2word.json"),raising=False)
    return tmp_path


@pytest.fixture
def use_real_custom_corpora(tmp_path, monkeypatch):
    """
    Copy the real `nn_preprocessing/custom/corpora/*` into tmp_path
    and override the module-level paths so that createTrainFile()
    reads from it and writes under tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    project_root = Path(__file__).parents[1]
    src = project_root /"data"/  "corpora" / "custom"
    dst = tmp_path  /"data"/  "corpora" / "custom"
    for name in ("casino_corpus", "deal_no_deal_corpus", "kodis_corpus"):
        shutil.copytree(src / name, dst / name)

    
    monkeypatch.setattr(data,   "repo_dir",    str(tmp_path), raising=False)
    monkeypatch.setattr(data,   "corpus_name", "custom",     raising=False)
    monkeypatch.setattr(data,   "corpus_dir",  str(dst),       raising=False)
    monkeypatch.setattr(cfg, "corpus_name", "custom",     raising=False)
    monkeypatch.setattr(cfg, "corpus_dir",  str(dst),       raising=False)
    monkeypatch.setattr(data, "data_dir",  str(tmp_path / "data"),       raising=False)

    monkeypatch.setattr(cfg, "word2index_path", str(dst / "word2index.json"),raising=False)
    monkeypatch.setattr(cfg, "index2word_path",str(dst / "index2word.json"),raising=False)
    return tmp_path

@pytest.fixture()
def use_real_train_data(monkeypatch):
    """
    Override cfg.repo_dir and cfg.corpus_name so that build_Vocab()
    reads your real `nn_input_data/custom/train_processed_dialogs.txt`.
    """
    project_root = Path(__file__).parent.parent.resolve()
    monkeypatch.setattr(cfg, "repo_dir", str(project_root), raising=False)
    # we want the 'custom' corpus that already exists under nn_input_data/custom
    monkeypatch.setattr(cfg, "corpus_name", "custom", raising=False)

def is_emoji(tok):
    return bool(regex.match(r'\p{Emoji}', tok)) and not tok.isdigit()

def detect_emojis(tok):
    return tok in emoji.EMOJI_DATA and not tok.isdigit()

def test_tokenize_basic():
    text = "Héllo, WORLD!"
    tokens = tokenize(text)
    assert tokens == ["hello", ",", "world", "!"]

@pytest.fixture
def fake_vocab_workspace(tmp_path, monkeypatch):
    # 1) choose a fake data root
    data_root = tmp_path / "data"
    # 2) create the input dir
    in_dir = data_root / "nn_input_data" / "mycorp"
    in_dir.mkdir(parents=True)

    # 3) copy your real train file there
    project_root = Path(__file__).parents[2]
    real_train = project_root /"src"/"data"/ "nn_input_data" / "custom" / "train_processed_dialogs_testing.txt"
    content = real_train.read_text()
    (in_dir / "train_processed_dialogs.txt").write_text(content)

    # 4) patch config so build_Vocab reads from this fake location
    monkeypatch.setattr(cfg, "data_dir", str(data_root), raising=False)
    monkeypatch.setattr(cfg, "corpus_name", "mycorp", raising=False)

    # 5) prepare the output dir and patch the output paths
    out_dir = data_root / "nn_preprocessing" / "mycorp"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(cfg,
        "word2index_path", str(out_dir / "word2index.json"), raising=False)
    monkeypatch.setattr(cfg,
        "index2word_path", str(out_dir / "index2word.json"), raising=False)
    return data_root

def test_tokenize_emojis():
    text = (
        "We need some firewood too, though! ☹️ Let's try to make a deal that "
        "benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?"
    )
    tokens = tokenize(text)
    escaped = [tok.encode("unicode_escape").decode("ascii") for tok in tokens]
    print("Tokenized (escaped):", escaped)
    assert "☹" in tokens
    assert "🙂" in tokens


def test_voc_add_and_trim_with_tokens():
    tokens = tokenize("hello world world")
    voc = Voc("test")
    for tok in tokens:
        voc.addWord(tok)
    assert voc.word2count["world"] == 2
    assert voc.num_words == 6
    voc.trim(min_count=2)
    assert "hello" not in voc.word2index
    assert "world" in voc.word2index
    assert voc.num_words == 5


def test_create_train_file(isolate_cwd, monkeypatch):
    # patch Corpus to return two dummy convos
    dialogs1 = [[DummyUtterance("a"), DummyUtterance("b")]]
    dialogs2 = [[DummyUtterance("c")]]
    convos = [DummyConversation(dialogs1), DummyConversation(dialogs2)]
    monkeypatch.setattr(data, "Corpus",
                        lambda filename: DummyCKCorpus.with_convos(convos),
                        raising=True)

    # run & assert
    monkeypatch.setattr(data, "data_dir", str(isolate_cwd / "data"), raising=False)
    data.createTrainFile()
    out_file = Path(isolate_cwd) / "data" / "nn_input_data" / "mycorp" / "train_processed_dialogs.txt"
    assert out_file.exists()
    lines = out_file.read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0]) == [{"text": "a"}, {"text": "b"}]
    assert json.loads(lines[1]) == [{"text": "c"}]


def test_process_dialog_handles_emojis_as_UNK():
    text = (
        "We need some firewood too, though! ☹️ Let's try to make a deal that "
        "benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?"
    )
    dialog = [{"text": text}]
    toks = tokenize(text)
    voc = Voc("emoji_test")
    for tok in toks:
        if detect_emojis(tok):
            print(f"Skipping emoji token: {tok}")
            continue
        voc.addWord(tok)
    processed = processDialog(voc, dialog)[0]
    tokens = processed["tokens"]
    assert tokens.count("UNK") == 2


def test_corpus_convo_integrity():
    project_root = Path(__file__).parents[1]
    src = project_root / "data" / "corpora" / "custom"
    for name in ("casino_corpus","deal_no_deal_corpus","kodis_corpus"):
        corpus = Corpus(filename=str(src/name))
        assert isinstance(corpus, Corpus)
        for convo in corpus.iter_conversations():
            assert convo.check_integrity(verbose=False)


def test_corpus_root_to_leaf_paths(monkeypatch):
    # override only config, so we can locate the kodis_corpus
    project_root = Path(__file__).parents[1]
    new_dir = project_root / "data" / "corpora" / "custom"
    monkeypatch.setattr(cfg, "corpus_dir", str(new_dir), raising=True)
    load_path = os.path.join(cfg.corpus_dir, "kodis_corpus")
    corpus = Corpus(filename=load_path)
    for convo in corpus.iter_conversations():
        paths = convo.get_root_to_leaf_paths()
        assert len(paths) == 1


@pytest.mark.usefixtures("use_real_custom_corpora")
def test_create_train_file_with_real_corpora(use_real_custom_corpora):
    # this test gets a temporary copy of the real corpora,
    # then calls createTrainFile() against it
    data.createTrainFile()
    out_file = Path(use_real_custom_corpora /"data" / "nn_input_data" / "custom" / "train_processed_dialogs.txt")
    assert out_file.exists()
    print("Train file written to:", out_file)
    lines = out_file.read_text().splitlines()
    print("Preview:\n", "\n".join(lines[:5]) + ("\n..." if len(lines)>5 else ""))

@pytest.mark.usefixtures("fake_vocab_workspace")
def test_build_vocab_over_real_custom_data(fake_vocab_workspace,capsys):
    build_Vocab()
    # out_dir = use_real_custom_corpora / "nn_preprocessing" / "custom" / "corpora"
    i2w = Path(cfg.index2word_path)
    w2i = Path(cfg.word2index_path)
    assert w2i.exists(), f"{w2i} was not created"
    assert i2w.exists(), f"{i2w} was not created"
    word2index = json.loads(w2i.read_text())
    index2word = json.loads(i2w.read_text())
    assert len(word2index) > 10, "Expected at least 10 entries in word2index.json"
    assert len(index2word) > 10, "Expected at least 10 entries in index2word.json"
    # print(f"\nVocabulary files written to: {out_dir}")
    sample = list(word2index.items())[:5]
    print("index2word file written to: ", i2w)
    print("First few entries of word2index:", sample)