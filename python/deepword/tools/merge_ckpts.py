from os import path
import glob

import tensorflow as tf
from scipy.stats import describe
from termcolor import colored
import fire

from deepword.hparams import load_hparams
from deepword.models.nlu_modeling import BertNLU
from deepword.utils import eprint


def merge_ckpt(ckpt1, ckpt2, fn_hparams1, fn_hparams2, save_to):
    # substitute some variables of ckpt1 with the same variables from ckpt2
    hp1 = load_hparams(fn_model_config=fn_hparams1)
    hp2 = load_hparams(fn_model_config=fn_hparams2)
    model = BertNLU.get_eval_model(hp1, device_placement="cpu")
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=10,
            save_relative_paths=True)
        saver.restore(sess, ckpt1)

        all_saved_vars = list(
            map(lambda v: v[0],
                tf.train.list_variables(ckpt2)))
        all_vars = tf.global_variables()
        freeze1 = set(hp1.bert_freeze_layers.split(","))
        freeze2 = set(hp2.bert_freeze_layers.split(","))

        allowed_to_restore = freeze1 - freeze2

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
        saver2.restore(sess, ckpt2)

        local_vars2 = sess.run(var_list)

        eprint("parameters that have changed:")
        for i in range(len(var_list)):
            eprint(var_list[i].op.name)
            eprint(describe(local_vars[i] - local_vars2[i], axis=None))

        eprint("save the merged model to {}".format(save_to))
        saver.save(sess, save_to)


def main(model_dir_to, model_dir_from):
    """
    load some parameters from `model_dir_from` to `model_dir_to` and save the
    new checkpoint to another dir under `model_dir_to` as "merged_last_weights".

    the changed parameters will be computed through hp.bert_freeze_layers of
    the two models.

    Only layers that are trained by `model_from` while frozen by `model_to` will
    be substituted.
    """

    best_ckpt2 = tf.train.latest_checkpoint(
        "{}/best_weights".format(model_dir_from))
    watched_files = path.join(
        model_dir_to, "last_weights", "after-epoch-*.index")
    files = [path.splitext(fn)[0] for fn in glob.glob(watched_files)]
    if len(files) == 0:
        eprint(colored("No checkpoint found!", "red"))
        return
    step2ckpt = dict(map(lambda fn: (int(fn.split("-")[-1]), fn), files))
    steps = sorted(list(step2ckpt.keys()))

    fn_hparams1 = "{}/hparams.json".format(model_dir_from)
    fn_hparams2 = "{}/hparams.json".format(model_dir_to)

    for step in steps:
        save_to = "{}/merged_last_weights/after-epoch-{}".format(
            model_dir_to, step)
        try:
            merge_ckpt(
                ckpt1=step2ckpt[step], ckpt2=best_ckpt2,
                fn_hparams1=fn_hparams1, fn_hparams2=fn_hparams2,
                save_to=save_to)
        except Exception as e:
            eprint("error for step {}:\n{}".format(step, e))


if __name__ == "__main__":
    fire.Fire(main)
