from os import path
import glob

import tensorflow as tf
from scipy.stats import describe
from termcolor import colored
import fire

from deepword.hparams import load_hparams
from deepword.models.nlu_modeling import BertNLU
from deepword.utils import eprint


def merge_ckpt(ckpt_to, ckpt_from, fn_hparams_to, fn_hparams_from, save_to):
    # substitute some variables of ckpt1 with the same variables from ckpt2
    hp_to = load_hparams(fn_model_config=fn_hparams_to)
    hp_from = load_hparams(fn_model_config=fn_hparams_from)
    model = BertNLU.get_eval_model(hp_to, device_placement="cpu")
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=10,
            save_relative_paths=True)
        saver.restore(sess, ckpt_to)

        all_saved_vars = list(
            map(lambda v: v[0],
                tf.train.list_variables(ckpt_from)))
        all_vars = tf.global_variables()
        # only frozen parameters can be substituted
        freeze_to = set(hp_to.bert_freeze_layers.split(","))
        freeze_from = set(hp_from.bert_freeze_layers.split(","))

        allowed_to_restore = freeze_to - freeze_from

        eprint("allowed to restore: {}".format(allowed_to_restore))

        var_list = [
            v for v in all_vars
            if (any([layer_name in v.op.name.split("/")
                     for layer_name in allowed_to_restore])
                and v.op.name in all_saved_vars)]

        local_vars = sess.run(var_list)

        eprint("the following vars will be restored: {}".format(
            "\n".join([v.op.name for v in var_list])))

        saver2 = tf.train.Saver(
            var_list=var_list, max_to_keep=10, save_relative_paths=True)
        saver2.restore(sess, ckpt_from)

        local_vars2 = sess.run(var_list)

        eprint("parameters that have changed:")
        for i in range(len(var_list)):
            eprint(var_list[i].op.name)
            eprint(describe(local_vars[i] - local_vars2[i], axis=None))

        eprint("save the merged model to {}".format(save_to))
        saver.save(sess, save_to)


def one_from_multiple_to(model_dir_to, model_dir_from):
    """
    load some parameters from `model_dir_from` to `model_dir_to` and save the
    new checkpoint to another dir under `model_dir_to` as "merged_last_weights".

    the changed parameters will be computed through hp.bert_freeze_layers of
    the two models.

    Only layers that are trained by `model_from` while frozen by `model_to` will
    be substituted.

    Scenario: (one ckpt-from, multiple ckpt-to)
        model_dir_to:
            - last_weights:
                - ckpt-to-1
                - ckpt-to-2
                - ...
            - hparams.json
            - (expected) merged_last_weights
                - ckpt-to-1
                - ckpt-to-2
                - ...
        model_dir_from:
            - best_weights/ckpt-from
            - hparams.json
    """

    best_ckpt_from = tf.train.latest_checkpoint(
        "{}/best_weights".format(model_dir_from))
    watched_files = path.join(
        model_dir_to, "last_weights", "after-epoch-*.index")
    files = [path.splitext(fn)[0] for fn in glob.glob(watched_files)]
    if len(files) == 0:
        eprint(colored("No checkpoint found!", "red"))
        return
    step2ckpt = dict(map(lambda fn: (int(fn.split("-")[-1]), fn), files))
    steps = sorted(list(step2ckpt.keys()))

    fn_hparams_to = "{}/hparams.json".format(model_dir_to)
    fn_hparams_from = "{}/hparams.json".format(model_dir_from)

    for step in steps:
        save_to = "{}/merged_last_weights/after-epoch-{}".format(
            model_dir_to, step)
        try:
            merge_ckpt(
                ckpt_to=step2ckpt[step], ckpt_from=best_ckpt_from,
                fn_hparams_to=fn_hparams_to, fn_hparams_from=fn_hparams_from,
                save_to=save_to)
        except Exception as e:
            eprint("error for step {}:\n{}".format(step, e))


def multiple_from_one_to(model_dir_to, model_dir_from):
    """
    load some parameters from `model_dir_from` to `model_dir_to` and save the
    new checkpoint to another dir under `model_dir_to` as "merged_last_weights".

    the changed parameters will be computed through hp.bert_freeze_layers of
    the two models.

    Only layers that are trained by `model_from` while frozen by `model_to` will
    be substituted.

    Scenario: (one ckpt-to, multiple ckpt-from)
        model_dir_to:
            - best_weights/ckpt-to
            - hparams.json
            - (expected) merged_last_weights
                - ckpt-to-1
                - ckpt-to-2
                - ...
        model_dir_from:
            - last_weights:
                - ckpt-from-1
                - ckpt-from-2
                - ...
            - hparams.json
    """

    best_ckpt_to = tf.train.latest_checkpoint(
        "{}/best_weights".format(model_dir_to))
    watched_files = path.join(
        model_dir_from, "last_weights", "after-epoch-*.index")
    files = [path.splitext(fn)[0] for fn in glob.glob(watched_files)]
    if len(files) == 0:
        eprint(colored("No checkpoint found!", "red"))
        return
    step2ckpt = dict(map(lambda fn: (int(fn.split("-")[-1]), fn), files))
    steps = sorted(list(step2ckpt.keys()))

    fn_hparams_to = "{}/hparams.json".format(model_dir_to)
    fn_hparams_from = "{}/hparams.json".format(model_dir_from)

    for step in steps:
        save_to = "{}/merged_last_weights/after-epoch-{}".format(
            model_dir_to, step)
        try:
            merge_ckpt(
                ckpt_to=best_ckpt_to, ckpt_from=step2ckpt[step],
                fn_hparams_to=fn_hparams_to, fn_hparams_from=fn_hparams_from,
                save_to=save_to)
        except Exception as e:
            eprint("error for step {}:\n{}".format(step, e))


def main(style, model_dir_to, model_dir_from):
    if style == 'one-from-multiple-to':
        one_from_multiple_to(model_dir_to, model_dir_from)
    elif style == 'multiple-from-one-to':
        multiple_from_one_to(model_dir_to, model_dir_from)
    else:
        pass


if __name__ == "__main__":
    fire.Fire(main)
