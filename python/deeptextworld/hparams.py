import json
import os
import sys
from collections import namedtuple
from os.path import join as pjoin
from typing import Optional, Dict, Any, Iterable

import ruamel.yaml
from tensorflow.contrib.training import HParams

dir_path = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
project_path = pjoin(dir_path, "../..")


class Conventions(namedtuple(
    "Conventions", (
        "bert_ckpt_dir",
        "bert_vocab_file",
        "nltk_vocab_file",
        "glove_vocab_file",
        "glove_emb_file",
        "albert_ckpt_dir",
        "albert_vocab_file",
        "albert_spm_path",
        "bert_cls_token",
        "bert_unk_token",
        "bert_padding_token",
        "bert_sep_token",
        "bert_mask_token",
        "bert_sos_token",
        "bert_eos_token",
        "albert_cls_token",
        "albert_unk_token",
        "albert_padding_token",
        "albert_sep_token",
        "albert_mask_token",
        "nltk_unk_token",
        "nltk_padding_token",
        "nltk_sos_token",
        "nltk_eos_token"))):
    pass


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
    bert_sos_token="[unused0]",
    bert_eos_token="[unused1]",
    albert_cls_token="[CLS]",
    albert_unk_token="<unk>",
    albert_padding_token="<pad>",
    albert_sep_token="[SEP]",
    albert_mask_token="[MASK]",
    nltk_unk_token="[UNK]",
    nltk_padding_token="[PAD]",
    nltk_sos_token="<S>",
    nltk_eos_token="</S>")


def get_model_hparams(model_creator: str) -> HParams:
    try:
        model_hparams = default_config[model_creator]
    except Exception as e:
        raise ValueError(
            'unknown model creator: {}\n{}'.format(model_creator, e))
    return model_hparams


default_config = {
    "default": HParams(
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
        n_tokens_per_action=10,
        n_actions=256,  # only works for DQN models
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
        collect_floor_plan=True,
        start_t_ignore_model_t=False,
        use_step_wise_reward=True,
        tokenizer_type="BERT",
        use_glove_emb=False,
        glove_emb_path="",
        glove_trainable=False,
        compute_policy_action_every_step=False),
    "LstmDQN": HParams(
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
    "CnnDQN": HParams(
        agent_clazz='BaseAgent',
        core_clazz="DQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "CnnDRRN": HParams(
        agent_clazz='BaseAgent',
        core_clazz="DRRNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "LegacyCnnDRRN": HParams(
        agent_clazz='BaseAgent',
        core_clazz="LegacyDRRNCore",
        tokenizer_type="NLTK",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=5,
        num_tokens=1000,
        num_conv_filters=32),
    "TransformerDRRN": HParams(
        agent_clazz='BaseAgent',
        core_clazz="DRRNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32),
    "BertDRRN": HParams(
        agent_clazz='BaseAgent',
        core_clazz="DRRNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=768,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=11,
        num_tokens=512,
        num_conv_filters=32,
        bert_num_hidden_layers=1,
        cls_val_id=0,
        sep_val_id=0,
        mask_val_id=0),
    "CnnDSQN": HParams(
        agent_clazz='DSQNAgent',
        core_clazz="DSQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=11,
        num_tokens=1000,
        num_conv_filters=32,
        snn_train_epochs=1000),
    "TransformerDSQN": HParams(
        agent_clazz='DSQNAgent',
        core_clazz="DSQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=6,
        num_tokens=500,
        num_conv_filters=32,
        snn_train_epochs=1000),
    "TransformerDSQNWithFactor": HParams(
        agent_clazz='DSQNAgent',
        core_clazz="DSQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        lstm_num_layers=1,
        num_turns=6,
        num_tokens=500,
        num_conv_filters=32,
        snn_train_epochs=1000),
    "TransformerGenDQN": HParams(
        agent_clazz='GenDQNAgent',
        core_clazz="GenDQNCore",
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=6,
        num_tokens=1000,
        max_decoding_size=10,
        decode_concat_action=False,
        tokenizer_type="NLTK"),
    "BertCommonsenseModel": HParams(
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
        mask_val_id=0
    ),
    "AlbertCommonsenseModel": HParams(
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
        mask_val_id=0
    )
}


def output_hparams(hp: HParams) -> str:
    out_str = ['------------hparams---------------']
    hp_dict = hp.values()
    keys = sorted(hp_dict.keys())
    for k in keys:
        out_str.append('{} -> {}'.format(k, hp_dict[k]))
    out_str.append('-----------------------------------')
    return "\n".join(out_str)


def update_hparams_from_dict(
        hp: HParams, cmd_args: Dict[str, Any],
        allowed_to_change: Optional[Iterable[str]] = None) -> HParams:
    for hp_key in cmd_args:
        if (hp_key in hp and cmd_args[hp_key] is not None and
                (hp_key in allowed_to_change if allowed_to_change else True)):
            hp.set_hparam(hp_key, cmd_args[hp_key])
    return hp


def update_hparams_from_hparams(hp: HParams, hp2: HParams) -> HParams:
    """hp should not have same keys with hp2"""
    dict_hp2 = hp2.values()
    for k in dict_hp2:
        if k not in hp:
            hp.add_hparam(k, dict_hp2.get(k))
        else:
            hp.set_hparam(k, dict_hp2.get(k))
    return hp


def update_hparams_from_file(hp: HParams, file_args: str) -> HParams:
    with open(file_args, 'r') as f:
        json_val = json.load(f)
        for k in json_val:
            if k in hp:
                hp.set_hparam(k, json_val.get(k))
            else:
                pass
    return hp


def copy_hparams(hp: HParams) -> HParams:
    hp2 = HParams()
    dict_hp = hp.values()
    for k in dict_hp:
        hp2.add_hparam(k, dict_hp.get(k))
    return hp2


def has_valid_val(dict_args: Optional[Dict[str, Any]], key: str) -> bool:
    """
    1. if dict_args exists
    2. if key in dict_args
    3. if dict_args[key] is not None
    """
    return (dict_args is not None and key in dict_args
            and dict_args[key] is not None)


def load_hparams(
        fn_model_config: Optional[str] = None,
        cmd_args: Optional[Dict[str, Any]] = None,
        fn_pre_config: Optional[str] = None) -> HParams:
    """
    load hyper-parameters
    priority(file_args) > priority(cmd_args) except arg in allowed_to_change
    priority(cmd_args) > priority(pre_config)
    priority(pre_config) > priority(default)
    """
    allowed_to_change = [
        "model_dir", "eval_episode", "game_episode_terminal_t",
        "batch_size", "learning_rate", "compute_policy_action_every_step",
        "max_snapshot_to_keep", "start_t_ignore_model_t", "annealing_eps_t",
        "collect_floor_plan", "init_eps", "final_eps", "save_gap_t"
    ]

    if fn_pre_config:
        with open(fn_pre_config, 'rt') as f:
            pre_config = ruamel.yaml.safe_load(f.read())
    else:
        pre_config = None

    hp = get_model_hparams("default")
    if fn_model_config is not None:
        hp = update_hparams_from_file(hp, fn_model_config)
        model_creator = hp.model_creator
    else:
        if (has_valid_val(cmd_args, "model_creator")
                and has_valid_val(pre_config, "model_creator")):
            assert cmd_args["model_creator"] == pre_config["model_creator"]
        model_creator = (
            cmd_args["model_creator"]
            if has_valid_val(cmd_args, "model_creator")
            else pre_config["model_creator"])

    model_hp = get_model_hparams(model_creator)
    hp = update_hparams_from_hparams(hp, model_hp)

    # second load hp from file to change params back
    if fn_model_config is not None:
        hp = update_hparams_from_file(hp, fn_model_config)
        if pre_config is not None:
            hp = update_hparams_from_dict(hp, pre_config, allowed_to_change)
        if cmd_args is not None:
            hp = update_hparams_from_dict(hp, cmd_args, allowed_to_change)
    elif cmd_args is not None:
        if pre_config is not None:
            hp = update_hparams_from_dict(hp, pre_config)
        hp = update_hparams_from_dict(hp, cmd_args)
    elif pre_config is not None:
        hp = update_hparams_from_dict(hp, pre_config)
    else:
        raise ValueError("file_args and cmd_args are both None")
    return hp


def save_hparams(hp: HParams, file_path: str) -> None:
    with open(file_path, 'w') as f:
        new_hp = copy_hparams(hp)
        new_hp.set_hparam("model_dir", ".")
        f.write(new_hp.to_json())
