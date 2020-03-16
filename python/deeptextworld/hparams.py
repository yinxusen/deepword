import json
import os
import sys
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Optional

import tensorflow as tf

from deeptextworld.utils import eprint

dir_path = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
project_path = pjoin(dir_path, "../..")


@dataclass(frozen=True)
class Conventions:
    bert_ckpt_dir: Optional[str]
    bert_vocab_file: Optional[str]
    nltk_vocab_file: Optional[str]
    glove_vocab_file: Optional[str]
    glove_emb_file: Optional[str]
    albert_ckpt_dir: Optional[str]
    albert_vocab_file: Optional[str]
    albert_spm_path: Optional[str]
    bert_cls_token: Optional[str]
    bert_unk_token: Optional[str]
    bert_padding_token: Optional[str]
    bert_sep_token: Optional[str]
    bert_mask_token: Optional[str]
    albert_cls_token: Optional[str]
    albert_unk_token: Optional[str]
    albert_padding_token: Optional[str]
    albert_sep_token: Optional[str]
    albert_mask_token: Optional[str]
    nltk_unk_token: Optional[str]
    nltk_padding_token: Optional[str]
    nltk_sos_token: Optional[str]
    nltk_eos_token: Optional[str]


conventions = Conventions(
    bert_ckpt_dir=pjoin(
        home_dir, "local/opt/bert-models/bert-model"),
    bert_vocab_file=pjoin(
        home_dir, "local/opt/bert-models/bert-model/vocab.txt"),
    nltk_vocab_file=pjoin(project_path, "resources/vocab.txt"),
    glove_vocab_file=pjoin(
        home_dir, "local/opt/glove-models/glove.6B/vocab.glove.6B.4more.txt"),
    glove_emb_file=pjoin(
        home_dir, "local/opt/glove-models/glove.6B/glove.6B.50d.4more.txt"),
    albert_ckpt_dir=pjoin(
        home_dir, "local/opt/bert-models/albert-model"),
    albert_vocab_file=pjoin(
        home_dir, "local/opt/bert-models/albert-model/30k-clean.vocab"),
    albert_spm_path=pjoin(
        home_dir, "local/opt/bert-models/albert-model/30k-clean.model"),
    bert_cls_token="[CLS]",
    bert_unk_token="[UNK]",
    bert_padding_token="[PAD]",
    bert_sep_token="[SEP]",
    bert_mask_token="[MASK]",
    albert_cls_token="[CLS]",
    albert_unk_token="<unk>",
    albert_padding_token="<pad>",
    albert_sep_token="[SEP]",
    albert_mask_token="[MASK]",
    nltk_unk_token="[UNK]",
    nltk_padding_token="[PAD]",
    nltk_sos_token="<S>",
    nltk_eos_token="</S>")


def get_model_hparams(model_creator):
    try:
        model_hparams = HPARAMS[model_creator]
    except Exception as e:
        raise ValueError(
            'unknown model creator: {}\n{}'.format(model_creator, e))
    return model_hparams


HPARAMS = {
    "default": tf.contrib.training.HParams(
        model_dir='',
        eval_episode=0,
        init_eps=1.,
        final_eps=1e-4,
        annealing_eps_t=5000,
        gamma=0.7,
        replay_mem=100000,
        observation_t=2000,
        total_t=sys.maxsize,
        game_episode_terminal_t=100,
        vocab_size=0,
        n_actions=128,
        n_tokens_per_action=10,
        hidden_state_size=32,
        sos=None,
        eos=None,
        padding_val=None,
        unk_val=None,
        cls_val=None,
        sep_val=None,
        mask_val=None,
        sos_id=None,
        eos_id=None,
        padding_val_id=None,
        unk_val_id=None,
        cls_val_id=None,
        sep_val_id=None,
        mask_val_id=None,
        model_creator='',
        max_snapshot_to_keep=5,
        jitter_go=False,
        jitter_eval_prob=1.,
        jitter_train_prob=0.5,
        collect_floor_plan=True,
        start_t_ignore_model_t=False,
        apply_dependency_parser=False,
        use_padding_over_lines=False,
        drop_w_theme_words=False,
        use_step_wise_reward=False,
        tokenizer_type="BERT",
        pad_eos=False,
        use_glove_emb=False,
        glove_emb_path="",
        glove_trainable=False),
    "LstmDQN": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="DQNCore",
        batch_size=32,
        save_gap_t=1000,
        lstm_num_units=32,
        lstm_num_layers=3,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000),
    "CnnDQN": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="DQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "CnnDRRN": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="DRRNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "LegacyCnnDRRN": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="LegacyDRRNCore",
        tokenizer_type="NLTK",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "TransformerDRRN": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="DRRNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "BertDRRN": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="DRRNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=768,
        learning_rate=1e-5,
        num_turns=11,
        num_tokens=512,
        num_conv_filters=32,
        bert_num_hidden_layers=1,
        cls_val_id=0,
        sep_val_id=0,
        mask_val_id=0),
    "CnnDSQN": tf.contrib.training.HParams(
        agent_clazz='DSQNAgent',
        core_clazz="DSQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32,
        snn_train_epochs=1000),
    "TransformerDSQN": tf.contrib.training.HParams(
        agent_clazz='DSQNAgent',
        core_clazz="DSQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=6,
        num_tokens=500,
        num_conv_filters=32,
        snn_train_epochs=1000),
    "TransformerDSQNWithFactor": tf.contrib.training.HParams(
        agent_clazz='DSQNAgent',
        core_clazz="DSQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=6,
        num_tokens=500,
        num_conv_filters=32,
        snn_train_epochs=1000),
    "TransformerGenDQN": tf.contrib.training.HParams(
        agent_clazz='GenDQNAgent',
        core_clazz="GenDQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=6,
        num_tokens=1000,
        max_action_len=10,
        tokenizer_type="NLTK",
        pad_eos=True),
    "BertCommonsenseModel": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="BertCore",
        batch_size=32,
        save_gap_t=1000,
        learning_rate=5e-5,
        num_turns=6,
        num_tokens=500,
        bert_num_hidden_layers=12,
        embedding_size=64,
        cls_val_id=0,
        sep_val_id=0,
        mask_val_id=0,
        n_classes=4  # for SWAG
    ),
    "AlbertCommonsenseModel": tf.contrib.training.HParams(
        agent_clazz='BaseAgent',
        core_clazz="BertCore",
        batch_size=32,
        save_gap_t=1000,
        learning_rate=5e-5,
        num_turns=6,
        num_tokens=500,
        bert_num_hidden_layers=12,
        embedding_size=64,
        padding_val_id=0,
        unk_val_id=0,
        cls_val_id=0,
        sep_val_id=0,
        mask_val_id=0,
        n_classes=4   # for SWAG
    )
}


def output_hparams(hp):
    out_str = ['------------hparams---------------']
    hp_dict = hp.values()
    keys = sorted(hp_dict.keys())
    for k in keys:
        out_str.append('{} -> {}'.format(k, hp_dict[k]))
    out_str.append('-----------------------------------')
    return "\n".join(out_str)


def update_hparams_from_cmd(hp, cmd_args):
    dict_cmd_args = vars(cmd_args)
    for hp_key in dict_cmd_args:
        if hp_key in hp and dict_cmd_args[hp_key] is not None:
            hp.set_hparam(hp_key, dict_cmd_args[hp_key])
    return hp


def update_hparams_from_hparams(hp, hp2):
    """hp should not have same keys with hp2"""
    dict_hp2 = hp2.values()
    for k in dict_hp2:
        if k not in hp:
            hp.add_hparam(k, dict_hp2.get(k))
        else:
            hp.set_hparam(k, dict_hp2.get(k))
    return hp


def update_hparams_from_file(hp, file_args):
    with open(file_args, 'r') as f:
        json_val = json.load(f)
        for k in json_val:
            if k in hp:
                hp.set_hparam(k, json_val.get(k))
            else:
                pass
    return hp


def copy_hparams(hp):
    hp2 = tf.contrib.training.HParams()
    dict_hp = hp.values()
    for k in dict_hp:
        hp2.add_hparam(k, dict_hp.get(k))
    return hp2


def load_hparams_for_training(file_args=None, cmd_args=None):
    """
    load hparams for training.
    priority(cmd_args) > priority(file_args)
    """
    hp = get_model_hparams("default")
    model_hp = get_model_hparams(cmd_args.model_creator)
    hp = update_hparams_from_hparams(hp, model_hp)
    if file_args is not None:
        hp = update_hparams_from_file(hp, file_args)
    if cmd_args is not None:
        hp = update_hparams_from_cmd(hp, cmd_args)
    return hp


def load_hparams_for_evaluation(pre_config_file, cmd_args=None):
    """
    load hparams for evaluation.
    priority(file_args) > priority(cmd_args)
     unless arg in allowed_to_change set.
    """
    allowed_to_change = [
        'model_dir', 'eval_episode', 'game_episode_terminal_t', "n_actions"]
    hp = get_model_hparams("default")
    # first load hp from file for choosing model_hp
    # notice that only hparams in hp can be updated.
    hp = update_hparams_from_file(hp, pre_config_file)
    model_hp = get_model_hparams(hp.model_creator)
    hp = update_hparams_from_hparams(hp, model_hp)
    # second load hp from file to change params back
    hp = update_hparams_from_file(hp, pre_config_file)

    if cmd_args is not None:
        dict_cmd_args = vars(cmd_args)
        for hp_key in dict_cmd_args:
            if (dict_cmd_args[hp_key] is not None
                    and hp_key in hp and hp_key in allowed_to_change):
                eprint("changing hparam {} ({} -> {})".format(
                    hp_key, hp.get(hp_key), dict_cmd_args[hp_key]))
                hp.set_hparam(hp_key, dict_cmd_args[hp_key])
            else:
                pass
    return hp


def save_hparams(hp, file_path):
    with open(file_path, 'w') as f:
        new_hp = copy_hparams(hp)
        new_hp.set_hparam("model_dir", ".")
        f.write(new_hp.to_json())
