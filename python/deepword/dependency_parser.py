from typing import Dict

from nltk import sent_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser

from deepword.log import Logging


class DependencyParserReorder(Logging):
    """
    Use dependency parser to reorder master sentences.
    Make sure to open Stanford CoreNLP server first.
    Refer to https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

    The DP reorder class is used with CNN layers for trajectory encoding.
    Refer to https://arxiv.org/abs/1905.02265 for details.
    """

    def __init__(self, padding_val: str, stride_len: int) -> None:
        """
        Args:
            padding_val: padding token, e.g. '[PAD]' or 'O'
            stride_len: CNN stride len
        """
        super(DependencyParserReorder, self).__init__()
        # be sure of starting CoreNLP server first
        self.parser = CoreNLPDependencyParser()
        # use dict to avoid parse the same sentences.
        self.parsed_sentences: Dict[str, str] = dict()
        self.sep_sent = (
                " " + " ".join([padding_val] * stride_len) + " ")

    def _reorder_sent(self, sent: str) -> str:
        """
        Use dependency parser to reorder a sentence.
        """
        tree = next(self.parser.raw_parse(sent)).tree()
        t_labels = ([
            [head.label()] +
            [child if type(child) is str else child.label() for child in head]
            for head in tree.subtrees()])
        t_str = [" ".join(labels) for labels in t_labels]
        return self.sep_sent.join(t_str)

    def _reorder_block(self, master: str) -> str:
        """
        Use dependency parser to reorder a paragraph.
        """
        sent_list = list(filter(lambda sent: sent != "", sent_tokenize(master)))
        tree_strs = []
        for s in sent_list:
            if s not in self.parsed_sentences:
                t_str = self._reorder_sent(s)
                self.parsed_sentences[s] = t_str
                self.info("parse {} into {}".format(s, t_str))
            else:
                self.info("found parsed {}".format(s))
            tree_strs.append(self.parsed_sentences[s])
        return self.sep_sent.join(tree_strs)

    def reorder(self, master: str) -> str:
        """
        Use dependency parser to reorder a paragraph.
        """
        if master == "":
            return master
        lines = map(lambda l: l.lower(),
                    filter(lambda l: l.strip() != "", master.split("\n")))
        reordered_lines = map(lambda l: self._reorder_block(l), lines)
        return (self.sep_sent + self.sep_sent.join(reordered_lines) +
                self.sep_sent)

