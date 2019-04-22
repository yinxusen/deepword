from bert.tokenization import FullTokenizer

from deeptextworld import bert_drrn_model
from deeptextworld.agents.drrn_agent import DRRNAgent
from deeptextworld.hparams import copy_hparams
from deeptextworld.utils import load_vocab, get_token2idx


class BertDRRNAgent(DRRNAgent):
    """
    """
    def __init__(self, hp, model_dir):
        super(BertDRRNAgent, self).__init__(hp, model_dir)
        self.tokenizer = FullTokenizer(
            vocab_file=hp.vocab_file, do_lower_case=True)

    def init_tokens(self, hp):
        """
        :param hp:
        :return:
        """
        new_hp = copy_hparams(hp)
        # make sure that padding_val is indexed as 0.
        tokens = list(load_vocab(hp.vocab_file))
        print(tokens[:10])
        token2idx = get_token2idx(tokens)
        new_hp.set_hparam('vocab_size', len(tokens))
        new_hp.set_hparam('sos_id', token2idx[hp.sos])
        new_hp.set_hparam('eos_id', token2idx[hp.eos])
        new_hp.set_hparam('padding_val_id', token2idx[hp.padding_val])
        new_hp.set_hparam('unk_val_id', token2idx[hp.unk_val])
        # bert specific tokens
        new_hp.set_hparam('cls_val_id', token2idx[hp.cls_val])
        new_hp.set_hparam('sep_val_id', token2idx[hp.sep_val])
        new_hp.set_hparam('mask_val_id', token2idx[hp.mask_val])
        return new_hp, tokens, token2idx

    def tokenize(self, master):
        return ' '.join([t.lower() for t in self.tokenizer.tokenize(master)])

    def create_model_instance(self):
        model_creator = getattr(bert_drrn_model, self.hp.model_creator)
        model = bert_drrn_model.create_train_model(model_creator, self.hp)
        return model

    def create_eval_model_instance(self):
        model_creator = getattr(bert_drrn_model, self.hp.model_creator)
        model = bert_drrn_model.create_eval_model(model_creator, self.hp)
        return model
