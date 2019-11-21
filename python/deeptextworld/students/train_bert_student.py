import glob
import os
import random
import sys
import time
from os.path import join as pjoin
from queue import Queue
from threading import Thread, Condition

import fire
import gym
import numpy as np
import tensorflow as tf
import textworld.gym
from numpy.random import choice as npc
from tqdm import trange

from deeptextworld import dsqn_model
from deeptextworld.action import ActionCollector
from deeptextworld.agents import dsqn_agent
from deeptextworld.agents.base_agent import BaseAgent, DRRNMemoTeacher
from deeptextworld.dsqn_model import create_train_student_drrn_model
from deeptextworld.hparams import load_hparams_for_training, output_hparams, \
    load_hparams_for_evaluation, save_hparams
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import flatten, eprint, ctime, setup_logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def prepare_model(fn_create_model, hp, device_placement, load_model_from):
    model_clazz = getattr(dsqn_model, hp.model_creator)
    model = fn_create_model(
        model_creator=model_clazz, hp=hp,
        device_placement=device_placement)
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=hp.max_snapshot_to_keep,
            save_relative_paths=True)
        global_step = tf.train.get_or_create_global_step()

    try:
        ckpt_path = tf.train.latest_checkpoint(load_model_from)
        saver.restore(sess, ckpt_path)
        trained_steps = sess.run(global_step)
        eprint("load student from ckpt: {}".format(ckpt_path))
    except Exception as e:
        eprint("load model failed: {}".format(e))
        trained_steps = 0
    return sess, model, saver, trained_steps


def train_bert_student(
        hp, tokenizer, model_path, combined_data_path, cond_of_eval):
    load_drrn_from = pjoin(model_path, "last_weights")
    ckpt_drrn_prefix = pjoin(load_drrn_from, "after-epoch")

    sess_drrn, model_drrn, saver_drrn, train_steps_drrn = prepare_model(
        create_train_student_drrn_model, hp, "/device:GPU:0", load_drrn_from)

    # save the very first model to verify the Bert weight has been loaded
    if train_steps_drrn == 0:
        saver_drrn.save(
            sess_drrn, ckpt_drrn_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model_drrn.graph))
    else:
        pass

    sw_path_drrn = pjoin(model_path, "drrn_summaries", "train")
    sw_drrn = tf.summary.FileWriter(sw_path_drrn, sess_drrn.graph)

    queue = Queue(maxsize=100)

    batch_size = hp.batch_size
    epoch_size = hp.save_gap_t
    num_epochs = 1000

    t = Thread(
        target=add_batch,
        args=(combined_data_path, queue, hp, batch_size, tokenizer))
    t.setDaemon(True)
    t.start()

    wait_times = 10
    while wait_times > 0 and queue.empty():
        eprint("waiting data ... (retry times: {})".format(wait_times))
        time.sleep(10)
        wait_times -= 1

    eprint("start training")
    data_in_queue = True
    for et in trange(num_epochs, ascii=True, desc="epoch"):
        for it in trange(epoch_size, ascii=True, desc="step"):
            try:
                data = queue.get(timeout=1)
                (p_states, p_len, action_matrix, action_mask_t, action_len,
                 expected_qs) = data
                _, summaries = sess_drrn.run(
                    [model_drrn.train_op, model_drrn.train_summary_op],
                    feed_dict={
                        model_drrn.src_: p_states,
                        model_drrn.src_len_: p_len,
                        model_drrn.actions_mask_: action_mask_t,
                        model_drrn.actions_: action_matrix,
                        model_drrn.actions_len_: action_len,
                        model_drrn.expected_qs_: expected_qs})
                sw_drrn.add_summary(
                    summaries, train_steps_drrn + et * epoch_size + it)
            except Exception as e:
                data_in_queue = False
                eprint("no more data: {}".format(e))
                break
        saver_drrn.save(
            sess_drrn, ckpt_drrn_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model_drrn.graph))
        eprint("finish and save {} epoch".format(et))
        with cond_of_eval:
            cond_of_eval.notifyAll()
        if not data_in_queue:
            break
    return


def clean_hs2tj(hash_states2tjs, tjs):
    cnt_trashed = 0
    empty_keys = []
    all_tids = list(tjs.trajectories.keys())
    for k in hash_states2tjs.keys():
        empty_tids = []
        for tid in hash_states2tjs[k].keys():
            if tid not in all_tids:
                empty_tids.append(tid)
                cnt_trashed += len(hash_states2tjs[k][tid])
        for tid in empty_tids:
            hash_states2tjs[k].pop(tid, None)
        if hash_states2tjs[k] == {}:  # delete the dict if empty
            empty_keys.append(k)
    eprint("hs2tj deletes {} items".format(cnt_trashed))
    for k in empty_keys:
        hash_states2tjs.pop(k, None)
    eprint("hs2tj deletes {} keys".format(len(empty_keys)))
    return hash_states2tjs


def add_batch(
        combined_data_path, queue, hp, batch_size, tokenizer):
    while True:
        for tp, ap, mp, hs in sorted(
                combined_data_path, key=lambda k: random.random()):
            memory, tjs, action_collector, hash_states2tjs = load_snapshot(
                hp, mp, tp, ap, hs, tokenizer)
            random.shuffle(memory)
            i = 0
            while i < len(memory) // batch_size:
                batch_memory = (
                    memory[i*batch_size: min((i+1)*batch_size, len(memory))])
                queue.put(
                    prepare_data(
                        batch_memory, tjs, action_collector, tokenizer,
                        hp.num_tokens),
                )
                i += 1


def load_snapshot(
        hp, memo_path, raw_tjs_path, action_path, hs2tj_path, tokenizer):
    memory = np.load(memo_path)['data']
    memory = list(filter(lambda x: isinstance(x, DRRNMemoTeacher), memory))

    tjs = RawTextTrajectory(hp)
    tjs.load_tjs(raw_tjs_path)

    actions = ActionCollector(
        tokenizer, hp.n_actions, hp.n_tokens_per_action,
        unk_val_id=hp.unk_val_id, padding_val_id=hp.padding_val_id)
    actions.load_actions(action_path)

    hs2tj = np.load(hs2tj_path)
    hash_states2tjs = hs2tj["hs2tj"][0]

    eprint(
        "snapshot data loaded:\nmemory path: {}\ntjs path:: {}\n"
        "action path: {}\nhs2tj path: {}".format(
            memo_path, raw_tjs_path, action_path, hs2tj_path))

    return memory, tjs, actions, hash_states2tjs


def prepare_master(master_str, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(master_str))


def prepare_action(action_str, tokenizer):
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(action_str))
    tokens = tokens[:max(10, len(tokens))]
    return tokens


def prepare_trajectory(trajectory_lst, tokenizer, num_tokens):
    rst_lst = flatten(
        [prepare_master(s, tokenizer) if i % 2 == 0 else
         prepare_action(s, tokenizer) for i, s in enumerate(trajectory_lst)])
    len_rst = len(rst_lst)
    if len_rst > num_tokens:
        rst_lst = rst_lst[len_rst-num_tokens:]
    else:
        rst_lst += [0] * (num_tokens - len_rst)
    len_rst = min(len_rst, num_tokens)
    return rst_lst, len_rst


def prepare_snn_pairs(batch_size, hash_states2tjs, tjs, num_tokens, tokenizer):
    non_empty_keys = list(
        filter(lambda x: hash_states2tjs[x] != {},
               hash_states2tjs.keys()))
    hs_keys = npc(non_empty_keys, size=batch_size)
    diff_keys = [
        npc(list(filter(lambda x: x != k, non_empty_keys)), size=None)
        for k in hs_keys]

    target_tids = []
    same_tids = []
    for k in hs_keys:
        try:
            tid_pair = npc(
                list(hash_states2tjs[k].keys()), size=2, replace=False)
        except ValueError:
            tid_pair = list(hash_states2tjs[k].keys()) * 2

        target_tids.append(tid_pair[0])
        same_tids.append(tid_pair[1])

    diff_tids = [npc(list(hash_states2tjs[k])) for k in diff_keys]

    target_sids = [npc(list(hash_states2tjs[k][tid]))
                   for k, tid in zip(hs_keys, target_tids)]
    same_sids = [npc(list(hash_states2tjs[k][tid]))
                 for k, tid in zip(hs_keys, same_tids)]
    diff_sids = [npc(list(hash_states2tjs[k][tid]))
                 for k, tid in zip(diff_keys, diff_tids)]

    target_state = tjs.fetch_batch_states(target_tids, target_sids)
    same_state = tjs.fetch_batch_states(same_tids, same_sids)
    diff_state = tjs.fetch_batch_states(diff_tids, diff_sids)

    target_src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in target_state]
    target_src = [x[0] for x in target_src_n_len]
    target_src_len = [x[1] for x in target_src_n_len]

    same_src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in same_state]
    same_src = [x[0] for x in same_src_n_len]
    same_src_len = [x[1] for x in same_src_n_len]

    diff_src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in diff_state]
    diff_src = [x[0] for x in diff_src_n_len]
    diff_src_len = [x[1] for x in diff_src_n_len]

    src = np.concatenate([target_src, target_src], axis=0)
    src_len = np.concatenate([target_src_len, target_src_len], axis=0)
    src2 = np.concatenate([same_src, diff_src], axis=0)
    src2_len = np.concatenate([same_src_len, diff_src_len], axis=0)
    labels = np.concatenate([np.zeros(batch_size), np.ones(batch_size)], axis=0)
    return src, src_len, src2, src2_len, labels


def prepare_data(b_memory, tjs, action_collector, tokenizer, num_tokens):
    trajectory_id = [m.tid for m in b_memory]
    state_id = [m.sid for m in b_memory]
    game_id = [m.gid for m in b_memory]
    action_mask = [m.action_mask for m in b_memory]
    expected_qs = [m.q_actions for m in b_memory]
    action_mask_t = BaseAgent.from_bytes(action_mask)

    states = tjs.fetch_batch_states(trajectory_id, state_id)
    states_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    p_states = list(map(lambda x: x[0], states_n_len))
    p_len = list(map(lambda x: x[1], states_n_len))

    action_len = (
        [action_collector.get_action_len(gid) for gid in game_id])
    max_action_len = np.max(action_len)
    action_matrix = (
        [action_collector.get_action_matrix(gid)[:, :max_action_len]
         for gid in game_id])

    return (
        p_states, p_len, action_matrix, action_mask_t, action_len, expected_qs)


def load_game_files(game_dir, f_games=None):
    """
    Choose games appearing in f_games in a given game dir. Return all games in
    the game dir if f_games is None.
    :param game_dir: a dir
    :param f_games: a file of game names
    :return: a list of games
    """
    if f_games is not None:
        with open(f_games, "r") as f:
            selected_games = map(lambda x: x.strip(), f.readlines())
        game_files = list(map(
            lambda x: os.path.join(game_dir, "{}.ulx".format(x)),
            selected_games))
    else:
        game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    return game_files


def split_train_dev(game_files):
    """
    Split train/dev sets from given game files
    sort - shuffle w/ Random(42) - 90%/10% split
      - if #game_files < 10, then use the last one as dev set;
      - if #game_files == 1, then use the one as both train and dev.
    :param game_files: game files
    :return: None if game_files is empty, otherwise (train, dev)
    """
    # have to sort first, otherwise after shuffling the result is different
    # on different platforms, e.g. Linux VS MacOS.
    game_files = sorted(game_files)
    random.Random(42).shuffle(game_files)
    if len(game_files) == 0:
        print("no game files found!")
        return None
    elif len(game_files) == 1:
        train_games = game_files
        dev_games = game_files
    elif len(game_files) < 10:  # use the last one as eval
        train_games = game_files[:-1]
        dev_games = game_files[-1:]
    else:
        num_train = int(len(game_files) * 0.9)
        train_games = game_files[:num_train]
        dev_games = game_files[num_train:]
    return train_games, dev_games


def run_agent_eval(
        agent, game_files, nb_episodes, max_episode_steps,
        snn_eval_data_size=100):
    """
    Run an eval agent on given games.
    :param agent:
    :param game_files:
    :param nb_episodes:
    :param max_episode_steps:
    :param snn_eval_data_size:
    :return:
    """
    eval_results = dict()
    requested_infos = agent.select_additional_infos()
    for game_no in range(len(game_files)):
        game_file = game_files[game_no]
        game_name = os.path.basename(game_file)
        env_id = textworld.gym.register_games(
            [game_file], requested_infos,
            max_episode_steps=max_episode_steps,
            name="eval")
        env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
        game_env = gym.make(env_id)
        eprint("eval game: {}".format(game_name))

        for episode_no in range(nb_episodes):
            action_list = []
            obs, infos = game_env.reset()
            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            while not all(dones):
                # Increase step counts.
                steps = ([step + int(not done)
                          for step, done in zip(steps, dones)])
                commands = agent.act(obs, scores, dones, infos)
                action_list.append(commands[0])
                obs, scores, dones, infos = game_env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            if game_name not in eval_results:
                eval_results[game_name] = []
            eval_results[game_name].append(
                (scores[0], infos["max_score"][0], steps[0],
                 infos["has_won"][0], action_list))
    # run snn eval after normal agent test
    # accuracy = agent.eval_snn(eval_data_size=snn_eval_data_size)
    accuracy = 0.
    return eval_results, accuracy


def agg_results(eval_results):
    """
    Aggregate evaluation results.
    :param eval_results:
    :return:
    """
    ret_val = {}
    total_scores = 0
    total_steps = 0
    all_scores = 0
    all_episodes = 0
    all_won = 0
    for game_id in eval_results:
        res = eval_results[game_id]
        agg_score = sum(map(lambda r: r[0], res))
        agg_max_score = sum(map(lambda r: r[1], res))
        all_scores += agg_max_score
        all_episodes += len(res)
        agg_step = sum(map(lambda r: r[2], res))
        agg_nb_won = len(list(filter(lambda r: r[3] , res)))
        all_won += agg_nb_won
        ret_val[game_id] = (agg_score, agg_max_score, agg_step, agg_nb_won)
        total_scores += agg_score
        total_steps += agg_step
    all_steps = all_episodes * 100
    return (ret_val, total_scores * 1. / all_scores,
            total_steps * 1. / all_steps, all_won * 1. / all_episodes)


def evaluation(hp, cv, model_dir, game_files, nb_episodes):
    """
    A thread of evaluation.
    :param hp:
    :param cv:
    :param model_dir:
    :param game_files:
    :param nb_episodes:
    :return:
    """
    eprint('evaluation worker started ...')
    eprint("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    eprint("games for eval: \n{}".format("\n".join(sorted(game_names))))

    agent_clazz = getattr(dsqn_agent, hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    # for eval during training, set load_best=False
    # agent.eval(load_best=False)

    prev_total_scores = 0
    prev_total_steps = sys.maxsize

    while True:
        with cv:
            cv.wait()
            eprint("start evaluation ...")
            agent.reset()
            eval_start_t = ctime()
            eval_results, snn_acc = run_agent_eval(
                agent, game_files, nb_episodes, hp.game_episode_terminal_t)
            eval_end_t = ctime()
            agg_res, total_scores, total_steps, n_won = agg_results(
                eval_results)
            eprint("eval_results: {}".format(eval_results))
            eprint("eval aggregated results: {}".format(agg_res))
            eprint(
                "after-epoch: {}, scores: {:.2f}, steps: {:.2f},"
                " n_won: {:.2f}".format(
                    agent.loaded_ckpt_step, total_scores, total_steps, n_won))
            eprint("SNN accuracy: {}".format(snn_acc))
            eprint(
                "time to finish eval: {}".format(eval_end_t-eval_start_t))
            if ((total_scores > prev_total_scores) or
                    ((total_scores == prev_total_scores) and
                     (total_steps < prev_total_steps))):
                eprint("found better agent, save model ...")
                prev_total_scores = total_scores
                prev_total_steps = total_steps
                agent.save_best_model()
            else:
                eprint("no better model, pass ...")


def train(cmd_args, combined_data_path, model_path, game_dir, f_games=None):
    hp = load_hparams_for_training(None, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    save_hparams(hp, pjoin(model_path, "hparams.json"))

    game_files = load_game_files(game_dir, f_games)
    games = split_train_dev(game_files)
    if games is None:
        exit(-1)
    train_games, dev_games = games
    cond_of_eval = Condition()
    eval_worker = Thread(
        name='eval_worker', target=evaluation,
        args=(hp, cond_of_eval, model_path, dev_games, hp.eval_episode))
    eval_worker.setDaemon(True)
    eval_worker.start()

    train_bert_student(
        hp, tokenizer, model_path, combined_data_path, cond_of_eval)


def run_eval(
        hp, model_dir, game_path, f_games=None, eval_randomness=None,
        eval_mode="eval-eval"):
    """
    Evaluation an agent.
    :param hp:
    :param model_dir:
    :param game_path:
    :param f_games:
    :param eval_randomness:
    :param eval_mode:
    :return:
    """
    if os.path.isdir(game_path):
        game_files = load_game_files(game_path, f_games)
        games = split_train_dev(game_files)
        if games is None:
            exit(-1)
        train_games, dev_games = games

        game_files = None
        if eval_mode == "all":
            # remove possible repeated games
            game_files = list(set(train_games + dev_games))
        elif eval_mode == "eval-train":
            game_files = train_games
        elif eval_mode == "eval-eval":
            game_files = dev_games
        else:
            eprint("unknown mode. choose from [all|eval-train|eval-eval]")
            exit(-1)
    elif os.path.isfile(game_path):
        game_files = [game_path]
    else:
        eprint("game path doesn't exist")
        return

    eprint("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    eprint("games for eval: \n{}".format("\n".join(sorted(game_names))))

    agent_clazz = getattr(dsqn_agent, hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.eval(load_best=False)
    if eval_randomness is not None:
        agent.eps = eval_randomness
    eprint("evaluation randomness: {}".format(agent.eps))

    eval_start_t = ctime()
    eval_results = run_agent_eval(
        agent, game_files, hp.eval_episode, hp.game_episode_terminal_t)
    eval_end_t = ctime()
    agg_res, total_scores, total_steps, n_won = agg_results(eval_results[0])
    eprint("eval_results: {}".format(eval_results[0]))
    eprint("eval aggregated results: {}".format(agg_res))
    eprint("scores: {:.2f}, steps: {:.2f}, n_won: {:.2f}".format(
        total_scores, total_steps, n_won))
    eprint("time to finish eval: {}".format(eval_end_t-eval_start_t))


def evaluate(cmd_args, model_path, game_path, f_games):
    config_file = pjoin(model_path, "hparams.json")
    hp = load_hparams_for_evaluation(config_file, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    run_eval(
        hp, model_path, game_path, f_games,
        eval_randomness=0,
        eval_mode="eval-eval")


def setup_train_log(model_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_config_file = '{}/../../../conf/logging.yaml'.format(current_dir)
    setup_logging(
        default_path=log_config_file,
        local_log_filename=os.path.join(model_dir, 'game_script.log'))


def main(data_path, n_data, model_path, game_path, f_games):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    setup_train_log(model_path)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.expanduser("~")
    bert_ckpt_dir = pjoin(home_dir, "local/opt/bert-models/bert-model")
    bert_vocab_file = pjoin(bert_ckpt_dir, "vocab.txt")
    nltk_vocab_file = pjoin(dir_path, "../resources/vocab.txt")

    tjs_prefix = "raw-trajectories"
    action_prefix = "actions"
    memo_prefix = "memo"
    hs2tj_prefix = "hs2tj"

    combined_data_path = []
    for i in sorted(range(n_data), key=lambda k: random.random()):
        combined_data_path.append(
            (pjoin(data_path, "{}-{}.npz".format(tjs_prefix, i)),
             pjoin(data_path, "{}-{}.npz".format(action_prefix, i)),
             pjoin(data_path, "{}-{}.npz".format(memo_prefix, i)),
             pjoin(data_path, "{}-{}.npz".format(hs2tj_prefix, i))))

    cmd_args = CMD(
        model_dir=model_path,
        model_creator="BertAttnEncoderDSQN",
        vocab_file=bert_vocab_file,
        bert_ckpt_dir=bert_ckpt_dir,
        num_tokens=511,
        num_turns=6,
        batch_size=32,
        save_gap_t=5000,
        embedding_size=64,
        learning_rate=5e-5,
        num_conv_filters=32,
        bert_num_hidden_layers=1,
        cls_val="[CLS]",
        cls_val_id=0,
        sep_val="[SEP]",
        sep_val_id=0,
        mask_val="[MASK]",
        mask_val_id=0,
        tokenizer_type="BERT",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000
    )

    train(cmd_args, combined_data_path, model_path, game_path, f_games)
    # evaluate(cmd_args, model_path, game_path, f_games)


if __name__ == "__main__":
    fire.Fire(main)
