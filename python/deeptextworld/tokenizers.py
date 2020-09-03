import string
from typing import List, Dict, Tuple

from albert.tokenization import FullTokenizer as AlbertTok
from bert.tokenization import FullTokenizer as BertTok
from nltk import word_tokenize
from tensorflow.contrib.training import HParams

from deeptextworld.hparams import conventions
from deeptextworld.hparams import copy_hparams
from deeptextworld.utils import load_vocab, get_token2idx


class Tokenizer(object):
    @property
    def vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @property
    def inv_vocab(self) -> Dict[int, str]:
        raise NotImplementedError()

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError()

    def de_tokenize(self, ids: List[int]) -> str:
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        raise NotImplementedError()


class NLTKTokenizer(Tokenizer):
    """
    Vocab is token2idx, inv_vocab is idx2token
    """

    def __init__(self, vocab_file, do_lower_case):
        self._special_tokens = [
            conventions.nltk_unk_token,
            conventions.nltk_padding_token,
            conventions.nltk_sos_token,
            conventions.nltk_eos_token]
        self._inv_vocab = load_vocab(vocab_file)
        if do_lower_case:
            self._inv_vocab = [
                w.lower() if w not in self._special_tokens else w
                for w in self._inv_vocab]
        self._do_lower_case = do_lower_case
        self._vocab = get_token2idx(self._inv_vocab)
        self._inv_vocab = dict([(v, k) for k, v in self._vocab.items()])
        self._unk_val_id = self._vocab[conventions.nltk_unk_token]
        self._s2c = {
            conventions.nltk_unk_token: "U",
            conventions.nltk_padding_token: "O",
            conventions.nltk_sos_token: "S",
            conventions.nltk_eos_token: "E"}
        self._c2s = dict(zip(self._s2c.values(), self._s2c.keys()))

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    def convert_tokens_to_ids(self, tokens):
        indexed = [self._vocab.get(t, self._unk_val_id) for t in tokens]
        return indexed

    def convert_ids_to_tokens(self, ids):
        tokens = [self._inv_vocab[i] for i in ids]
        return tokens

    def tokenize(self, text):
        if any([sc in text for sc in self._special_tokens]):
            new_txt = text
            for sc in self._special_tokens:
                new_txt = new_txt.replace(sc, self._s2c[sc])
            tokens = word_tokenize(new_txt)
            tokens = [self._c2s[t] if t in self._c2s else t for t in tokens]
        else:
            tokens = word_tokenize(text)

        if self._do_lower_case:
            return [
                t.lower() if t not in self._special_tokens else t
                for t in tokens]
        else:
            return tokens

    def de_tokenize(self, ids: List[int]) -> str:
        res = " ".join(
            filter(lambda t: t not in self._special_tokens,
                   self.convert_ids_to_tokens(ids)))
        return res


class LegacyZorkTokenizer(NLTKTokenizer):
    def __init__(self, vocab_file):
        super(LegacyZorkTokenizer, self).__init__(
            vocab_file, do_lower_case=True)
        self.empty_trans_table = str.maketrans("", "", string.punctuation)

    def tokenize(self, text):
        tokens = super(LegacyZorkTokenizer, self).tokenize(
            text.translate(self.empty_trans_table))
        return list(filter(lambda t: t.isalpha(), tokens))


class BertTokenizer(Tokenizer):
    def __init__(self, vocab_file, do_lower_case):
        self.tokenizer = BertTok(vocab_file, do_lower_case)
        self._special_tokens = [
            conventions.bert_unk_token,
            conventions.bert_padding_token,
            conventions.bert_cls_token,
            conventions.bert_sep_token,
            conventions.bert_mask_token,
            conventions.bert_sos_token,
            conventions.bert_eos_token]

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def de_tokenize(self, ids):
        res = " ".join(
            filter(lambda t: t not in self._special_tokens,
                   self.convert_ids_to_tokens(ids)))
        res = res.replace(" ##", "")
        return res

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)


class AlbertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case, spm_model_file):
        super(AlbertTokenizer, self).__init__(vocab_file, do_lower_case)
        self.tokenizer = AlbertTok(vocab_file, do_lower_case, spm_model_file)
        self._special_tokens = [
            conventions.albert_unk_token,
            conventions.albert_padding_token,
            conventions.albert_cls_token,
            conventions.albert_sep_token,
            conventions.albert_mask_token]

    def de_tokenize(self, ids):
        res = " ".join(
            filter(lambda t: t not in self._special_tokens,
                   self.convert_ids_to_tokens(ids)))
        res = res.replace(u"\u2581", " ")
        return res


def get_bert_tokenizer(hp: HParams) -> Tuple[HParams, Tokenizer]:
    tokenizer = BertTokenizer(
        vocab_file=conventions.bert_vocab_file, do_lower_case=True)
    new_hp = copy_hparams(hp)
    # set vocab info
    new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
    new_hp.set_hparam("padding_val", conventions.bert_padding_token)
    new_hp.set_hparam("unk_val", conventions.bert_unk_token)
    new_hp.set_hparam("cls_val", conventions.bert_cls_token)
    new_hp.set_hparam("sep_val", conventions.bert_sep_token)
    new_hp.set_hparam("mask_val", conventions.bert_mask_token)
    new_hp.set_hparam("sos", conventions.bert_sos_token)
    new_hp.set_hparam("eos", conventions.bert_eos_token)

    # set special token ids
    new_hp.set_hparam(
        'padding_val_id', tokenizer.vocab[conventions.bert_padding_token])
    assert new_hp.padding_val_id == 0, "padding should be indexed as 0"
    new_hp.set_hparam(
        'unk_val_id', tokenizer.vocab[conventions.bert_unk_token])
    # bert specific tokens
    new_hp.set_hparam(
        'cls_val_id', tokenizer.vocab[conventions.bert_cls_token])
    new_hp.set_hparam(
        'sep_val_id', tokenizer.vocab[conventions.bert_sep_token])
    new_hp.set_hparam(
        'mask_val_id', tokenizer.vocab[conventions.bert_mask_token])
    new_hp.set_hparam(
        "sos_id", tokenizer.vocab[conventions.bert_sos_token])
    new_hp.set_hparam(
        "eos_id", tokenizer.vocab[conventions.bert_eos_token])
    return new_hp, tokenizer


def get_albert_tokenizer(hp: HParams) -> Tuple[HParams, Tokenizer]:
    tokenizer = AlbertTokenizer(
        vocab_file=conventions.albert_vocab_file,
        do_lower_case=True,
        spm_model_file=conventions.albert_spm_path)
    new_hp = copy_hparams(hp)
    # make sure that padding_val is indexed as 0.
    new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
    new_hp.set_hparam("padding_val", conventions.albert_padding_token)
    new_hp.set_hparam("unk_val", conventions.albert_unk_token)
    new_hp.set_hparam("cls_val", conventions.albert_cls_token)
    new_hp.set_hparam("sep_val", conventions.albert_sep_token)
    new_hp.set_hparam("mask_val", conventions.albert_mask_token)

    new_hp.set_hparam(
        'padding_val_id', tokenizer.vocab[conventions.albert_padding_token])
    assert new_hp.padding_val_id == 0, "padding should be indexed as 0"
    new_hp.set_hparam(
        'unk_val_id', tokenizer.vocab[conventions.albert_unk_token])
    new_hp.set_hparam(
        'cls_val_id', tokenizer.vocab[conventions.albert_cls_token])
    new_hp.set_hparam(
        'sep_val_id', tokenizer.vocab[conventions.albert_sep_token])
    new_hp.set_hparam(
        'mask_val_id', tokenizer.vocab[conventions.albert_mask_token])
    return new_hp, tokenizer


def get_nltk_tokenizer(
        hp: HParams, vocab_file: str = conventions.nltk_vocab_file
) -> Tuple[HParams, Tokenizer]:
    tokenizer = NLTKTokenizer(vocab_file=vocab_file, do_lower_case=True)
    new_hp = copy_hparams(hp)
    new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
    new_hp.set_hparam("padding_val", conventions.nltk_padding_token)
    new_hp.set_hparam("unk_val", conventions.nltk_unk_token)
    new_hp.set_hparam("sos", conventions.nltk_sos_token)
    new_hp.set_hparam("eos", conventions.nltk_eos_token)

    new_hp.set_hparam(
        'padding_val_id', tokenizer.vocab[conventions.nltk_padding_token])
    assert new_hp.padding_val_id == 0, "padding should be indexed as 0"
    new_hp.set_hparam(
        'unk_val_id', tokenizer.vocab[conventions.nltk_unk_token])
    new_hp.set_hparam('sos_id', tokenizer.vocab[conventions.nltk_sos_token])
    new_hp.set_hparam('eos_id', tokenizer.vocab[conventions.nltk_eos_token])
    return new_hp, tokenizer


def get_zork_tokenizer(
        hp: HParams,
        vocab_file: str = conventions.legacy_zork_vocab_file
) -> Tuple[HParams, Tokenizer]:
    tokenizer = LegacyZorkTokenizer(vocab_file=vocab_file)
    new_hp = copy_hparams(hp)
    new_hp.set_hparam('vocab_size', len(tokenizer.vocab))
    new_hp.set_hparam("padding_val", conventions.nltk_padding_token)
    new_hp.set_hparam("unk_val", conventions.nltk_unk_token)
    new_hp.set_hparam("sos", conventions.nltk_sos_token)
    new_hp.set_hparam("eos", conventions.nltk_eos_token)

    new_hp.set_hparam(
        'padding_val_id', tokenizer.vocab[conventions.nltk_padding_token])
    new_hp.set_hparam(
        'unk_val_id', tokenizer.vocab[conventions.nltk_unk_token])
    new_hp.set_hparam('sos_id', tokenizer.vocab[conventions.nltk_sos_token])
    new_hp.set_hparam('eos_id', tokenizer.vocab[conventions.nltk_eos_token])
    return new_hp, tokenizer


def init_tokens(hp: HParams) -> Tuple[HParams, Tokenizer]:
    """
    Note that BERT must use bert vocabulary.
    :param hp:
    :return:
    """
    if hp.tokenizer_type.lower() == "bert":
        new_hp, tokenizer = get_bert_tokenizer(hp)
    elif hp.tokenizer_type.lower() == "albert":
        new_hp, tokenizer = get_albert_tokenizer(hp)
    elif hp.tokenizer_type.lower() == "nltk":
        if hp.use_glove_emb:
            # the glove vocab file has been modified to have special tokens
            # i.e. [PAD] [UNK] <S> </S>
            new_hp, tokenizer = get_nltk_tokenizer(
                hp, vocab_file=conventions.glove_vocab_file)
        else:
            new_hp, tokenizer = get_nltk_tokenizer(hp)
    elif hp.tokenizer_type.lower() == "zork":
        new_hp, tokenizer = get_zork_tokenizer(hp)
    else:
        raise ValueError(
            "Unknown tokenizer type: {}".format(hp.tokenizer_type))
    return new_hp, tokenizer
