from nltk import sent_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser

from deeptextworld.log import Logging


class DependencyParserReorder(Logging):
    """
    Use dependency parser to reorder master sentences.
    Make sure to open Stanford CoreNLP server first.
    """
    def __init__(self, padding_val, stride_len):
        super(DependencyParserReorder, self).__init__()
        # be sure of starting CoreNLP server first
        self.parser = CoreNLPDependencyParser()
        # use dict to avoid parse the same sentences.
        self.parsed_sentences = dict()
        self.sep_sent = (" " + " ".join([padding_val] * stride_len)
                         + " ")

    def reorder_sent(self, sent):
        tree = next(self.parser.raw_parse(sent)).tree()
        t_labels = ([
            [head.label()] +
            [child if type(child) is str else child.label() for child in head]
            for head in tree.subtrees()])
        t_str = [" ".join(labels) for labels in t_labels]
        return self.sep_sent.join(t_str)

    def reorder_block(self, master):
        """
        Notice that four padding letters " O O O O " can only work up to 5-gram
        :param master:
        :return:
        """
        sent_list = list(filter(lambda sent: sent != "", sent_tokenize(master)))
        tree_strs = []
        for s in sent_list:
            if s not in self.parsed_sentences:
                t_str = self.reorder_sent(s)
                self.parsed_sentences[s] = t_str
                self.info("parse {} into {}".format(s, t_str))
            else:
                self.info("found parsed {}".format(s))
            tree_strs.append(self.parsed_sentences[s])
        return self.sep_sent.join(tree_strs)

    def reorder(self, master):
        if master == "":
            return master
        lines = map(lambda l: l.lower(),
                    filter(lambda l: l.strip() != "", master.split("\n")))
        reordered_lines = map(lambda l: self.reorder_block(l), lines)
        return (self.sep_sent + self.sep_sent.join(reordered_lines) +
                self.sep_sent)

