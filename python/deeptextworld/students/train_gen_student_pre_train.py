import glob
import os
import random
import sys
import time
from os.path import join as pjoin
from queue import Queue
from threading import Thread

import fire
import gym
import numpy as np
import tensorflow as tf
import textworld.gym
from tqdm import trange

from deeptextworld import dqn_model
from deeptextworld.action import ActionCollector
from deeptextworld.agents import dqn_agent
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.dqn_model import create_train_gen_model
from deeptextworld.hparams import load_hparams_for_training, output_hparams, \
    load_hparams_for_evaluation, save_hparams
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import ctime, setup_logging
from deeptextworld.utils import flatten, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def prepare_model(fn_create_model, hp, device_placement, load_model_from):
    model_clazz = getattr(dqn_model, hp.model_creator)
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


def train_gen_student(
        hp, tokenizer, model_path, combined_data_path):
    load_from = pjoin(model_path, "last_weights")
    ckpt_prefix = pjoin(load_from, "after-epoch")

    sess, model, saver, train_steps = prepare_model(
        create_train_gen_model, hp, "/device:GPU:0", load_from)

    # save the very first model to verify weight has been loaded
    if train_steps == 0:
        saver.save(
            sess, ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model.graph))
    else:
        pass

    sw_path = pjoin(model_path, "summaries", "train")
    sw = tf.summary.FileWriter(sw_path, sess.graph)

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
                data = queue.get(timeout=10)
                (p_states, p_len, actions_in, actions_out, action_len,
                 expected_qs, b_weights) = data
                eprint(p_states)
                eprint(actions_in)
                _, summaries = sess.run(
                    [model.train_seq2seq_op, model.train_seq2seq_summary_op],
                    feed_dict={model.src_: p_states,
                               model.src_len_: p_len,
                               model.action_idx_: actions_in,
                               model.action_idx_out_: actions_out,
                               model.action_len_: action_len,
                               model.b_weight_: b_weights})
                sw.add_summary(
                    summaries, train_steps + et * epoch_size + it)
            except Exception as e:
                data_in_queue = False
                eprint("no more data: {}".format(e))
                break
        saver.save(
            sess, ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model.graph))
        eprint("finish and save {} epoch".format(et))
        if not data_in_queue:
            break
    return


def q_idx_to_action(q_idx, valid_len, tokens):
    """
    :param q_idx:
    :param valid_len: length includes </S>
    :param tokens:
    :return:
    """
    if valid_len <= 1:
        return " "
    return " ".join(map(lambda t: tokens[t], q_idx[:valid_len-1]))


def add_batch(
        combined_data_path, queue, hp, batch_size, tokenizer):
    while True:
        for tp, ap, mp, hs in sorted(
                combined_data_path, key=lambda k: random.random()):
            memory, tjs, action_collector = load_snapshot(
                hp, mp, tp, ap, tokenizer)
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


def load_snapshot(hp, memo_path, raw_tjs_path, action_path, tokenizer):
    memory = np.load(memo_path)['data']
    memory = list(filter(lambda x: isinstance(x, tuple), memory))

    tjs = RawTextTrajectory(hp)
    tjs.load_tjs(raw_tjs_path)

    actions = ActionCollector(
        tokenizer, hp.n_actions, hp.n_tokens_per_action,
        unk_val_id=hp.unk_val_id, padding_val_id=hp.padding_val_id)
    actions.load_actions(action_path)
    return memory, tjs, actions


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


def get_action_idx_pair(action_matrix, action_len, sos_id, eos_id):
    action_id_in = np.concatenate(
        [np.asarray([[sos_id]] * len(action_len)),
         action_matrix[:, :-1]], axis=1)
    action_id_out = action_matrix[:, :]
    n_rows, max_col_size = action_matrix.shape
    new_action_len = np.min(
        [action_len + 1, np.zeros_like(action_len) + max_col_size], axis=0)
    action_id_out[list(range(n_rows)), new_action_len-1] = eos_id
    return action_id_in, action_id_out, new_action_len


def get_q_per_token(action_idx, expected_qs, vocab_size):
    """
    :param action_idx: action idx after masking
    :param expected_qs: expected_qs after masking
    :return:
    """
    n_cols = action_idx.shape[1]
    n_rows = vocab_size
    n_actions = action_idx.shape[0]
    expected_q_mat = np.full([n_rows, n_cols], fill_value=-np.inf)
    for i in range(n_cols):
        for k in n_actions:
            if expected_qs[k] > expected_q_mat[action_idx[k, i], i]:
                expected_q_mat[action_idx[k, i], i] = expected_qs[k]
            else:
                pass
    return expected_q_mat


def get_best_q_idx(expected_qs, mask_idx):
    best_q_idx = []
    for i, mk_id in enumerate(mask_idx):
        bq_idx = mk_id[np.argmax(expected_qs[i][mk_id])]
        best_q_idx.append(bq_idx)
    return best_q_idx


def prepare_data(b_memory, tjs, action_collector, tokenizer, num_tokens):
    """
    ("tid", "sid", "gid", "aid", "reward", "is_terminal",
     "action_mask", "next_action_mask", "q_actions")
    """
    trajectory_id = [m[0] for m in b_memory]
    state_id = [m[1] for m in b_memory]
    game_id = [m[2] for m in b_memory]
    action_mask = [m[6] for m in b_memory]
    expected_qs = [m[8] for m in b_memory]
    action_mask_t = list(BaseAgent.from_bytes(action_mask))
    selected_mask_idx = list(map(
        lambda m: np.random.choice(np.where(m == 1)[0], size=[2, ]),
        action_mask_t))

    states = tjs.fetch_batch_states(trajectory_id, state_id)
    states_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    p_states = list(map(lambda x: x[0], states_n_len))
    p_len = list(map(lambda x: x[1], states_n_len))

    action_len = np.concatenate(
        [action_collector.get_action_len(gid)[mid]
         for gid, mid in zip(game_id, selected_mask_idx)], axis=0)
    actions = np.concatenate(
        [action_collector.get_action_matrix(gid)[mid, :]
         for gid, mid in zip(game_id, selected_mask_idx)], axis=0)
    actions_in, actions_out, action_len = get_action_idx_pair(
        actions, action_len, tokenizer.vocab["<S>"], tokenizer.vocab["</S>"])
    # repeats = np.sum(action_mask_t, axis=1)
    repeats = 2
    repeated_p_states = np.repeat(p_states, repeats, axis=0)
    repeated_p_len = np.repeat(p_len, repeats, axis=0)
    expected_qs = np.concatenate(
        [qs[mid] for qs, mid in zip(expected_qs, selected_mask_idx)], axis=0)
    b_weights = np.ones_like(action_len, dtype="float32")
    return (
        repeated_p_states, repeated_p_len,
        actions_in, actions_out, action_len, expected_qs, b_weights)


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
        agent, game_files, nb_episodes, max_episode_steps):
    """
    Run an eval agent on given games.
    :param agent:
    :param game_files:
    :param nb_episodes:
    :param max_episode_steps:
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
    return eval_results


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

    agent_clazz = getattr(dqn_agent, hp.agent_clazz)
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
            eval_results = run_agent_eval(
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


def train(cmd_args, combined_data_path, model_path):
    hp = load_hparams_for_training(None, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    save_hparams(hp, pjoin(model_path, "hparams.json"))

    train_gen_student(
        hp, tokenizer, model_path, combined_data_path)


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

    agent_clazz = getattr(dqn_agent, hp.agent_clazz)
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
    eprint("eval_results: {}".format(eval_results))
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


def main(data_path, n_data, model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    setup_train_log(model_path)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.expanduser("~")
    project_path = pjoin(dir_path, "../../..")
    bert_ckpt_dir = pjoin(home_dir, "local/opt/bert-models/bert-model")
    bert_vocab_file = pjoin(bert_ckpt_dir, "vocab.txt")
    nltk_vocab_file = pjoin(project_path, "resources/vocab.txt")

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
        model_creator="AttnEncoderDecoderDQN",
        vocab_file=nltk_vocab_file,
        num_tokens=512,
        num_turns=6,
        batch_size=32,
        save_gap_t=50000,
        embedding_size=64,
        learning_rate=5e-5,
        tokenizer_type="NLTK",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True
    )

    train(cmd_args, combined_data_path, model_path)
    # evaluate(cmd_args, model_path, game_path, f_games)


if __name__ == "__main__":
    fire.Fire(main)
