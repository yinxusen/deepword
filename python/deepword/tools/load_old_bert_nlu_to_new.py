import glob
from os import path

import fire
import tensorflow as tf
from termcolor import colored

from deepword.hparams import load_hparams
from deepword.models.nlu_modeling import BertNLU
from deepword.utils import eprint


def new_name2old_name(v_name):
    """
    This function is a very specific treatment for substituting
     q-encoder/xxx/yyy into bert-state-encoder/xxx/yyy
    """
    return "/".join(['bert-state-encoder'] + v_name.split("/")[1:])


def load_old_ckpt_to_new(ckpt, fn_hparams, save_to):
    # substitute some variables of ckpt1 with the same variables from ckpt2
    hp = load_hparams(fn_model_config=fn_hparams)
    model = BertNLU.get_eval_model(hp, device_placement="cpu")
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)

    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        all_saved_vars = list(
            map(lambda v: v[0],
                tf.train.list_variables(ckpt)))
        all_vars = tf.global_variables()
        var_list = [
            (v.op.name, v) for v in all_vars if v.op.name in all_saved_vars]
        extra_vars = [
            (new_name2old_name(v.op.name), v)
            for v in all_vars if "q-encoder" in v.op.name]

        name2var = dict(var_list + extra_vars)

        model_init_status = []
        for v_name, v in extra_vars:
            local_var = sess.run(name2var[v_name])
            model_init_status.append((v_name, v, local_var))

        saver2 = tf.train.Saver(var_list=name2var)
        saver2.restore(sess, ckpt)

        for v_name, v, local_init_var in model_init_status:
            local_var = sess.run(name2var[v_name])
            print(v_name)
            print(local_var)

        saver = tf.train.Saver(save_relative_paths=True)
        saver.save(sess, save_to)


def main(model_dir):
    """
    load some parameters from `model_dir_from` to `model_dir_to` and save the
    new checkpoint to another dir under `model_dir_to` as "merged_last_weights".

    the changed parameters will be computed through hp.bert_freeze_layers of
    the two models.

    Only layers that are trained by `model_from` while frozen by `model_to` will
    be substituted.
    """

    watched_files = path.join(
        model_dir, "last_weights", "after-epoch-*.index")
    files = [path.splitext(fn)[0] for fn in glob.glob(watched_files)]
    if len(files) == 0:
        eprint(colored("No checkpoint found!", "red"))
        return
    step2ckpt = dict(map(lambda fn: (int(fn.split("-")[-1]), fn), files))
    steps = sorted(list(step2ckpt.keys()))

    fn_hparams = "{}/hparams.json".format(model_dir)

    for step in steps:
        save_to = "{}/new_last_weights/after-epoch-{}".format(
            model_dir, step)
        try:
            load_old_ckpt_to_new(
                ckpt=step2ckpt[step],
                fn_hparams=fn_hparams,
                save_to=save_to)
        except Exception as e:
            eprint("error for step {}:\n{}".format(step, e))


if __name__ == "__main__":
    fire.Fire(main)
