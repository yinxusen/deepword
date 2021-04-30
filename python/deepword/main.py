import logging
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from typing import Optional, Callable

import gym
import tensorflow as tf
import textworld.gym
from gym.core import Env
from tensorflow.contrib.training import HParams
from termcolor import colored
from tqdm import trange
from alfworld.agents.utils.misc import Demangler

from deepword.agents.base_agent import BaseAgent
from deepword.eval_games import MultiGPUsEvalPlayer, LoopDogEvalPlayer, \
    FullDirEvalPlayer, agent_collect_data, agent_collect_data_v2
from deepword.eval_games import list_checkpoints
from deepword.hparams import load_hparams, conventions
from deepword.utils import agent_name2clazz, learner_name2clazz
from deepword.utils import load_game_files, load_alfworld_games
from deepword.utils import setup_train_log, setup_eval_log, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


def hp_parser() -> ArgumentParser:
    """
    Arg parser for hyper-parameters
    """

    # TODO: "store_true" defaults to be False, so use an explict default=None
    parser = ArgumentParser(argument_default=None)
    parser.add_argument('--model-creator', type=str)
    parser.add_argument('--agent-clazz', type=str)
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--init-eps', type=float)
    parser.add_argument('--final-eps', type=float)
    parser.add_argument('--annealing-eps-t', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--eval-episode', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--save-gap-t', type=int)
    parser.add_argument('--replay-mem', type=int)
    parser.add_argument('--observation-t', type=int)
    parser.add_argument('--total-t', default=sys.maxsize, type=int)
    parser.add_argument('--game-episode-terminal-t', type=int)
    parser.add_argument(
        '--collect-floor-plan', action='store_true', default=None)
    parser.add_argument(
        '--disable-collect-floor-plan', dest='collect_floor_plan',
        action='store_false')
    parser.add_argument(
        '--start-t-ignore-model-t', action='store_true', default=None)
    parser.add_argument('--n-actions', type=int)
    parser.add_argument(
        '--use-step-wise-reward', action='store_true', default=None)
    parser.add_argument(
        '--always-compute-policy', action='store_true', default=None)
    parser.add_argument("--tokenizer-type", type=str, help="[bert|albert|nltk]")
    parser.add_argument("--max-snapshot-to-keep", type=int)
    parser.add_argument(
        "--policy-to-action", type=str, help="[EPS/LinUCB/Sampling]")
    parser.add_argument("--policy-sampling-temp", type=float)
    parser.add_argument("--action-file", type=str)
    parser.add_argument("--policy-eps", type=float)
    parser.add_argument("--bert-language-layer", type=int)
    parser.add_argument("--bert-freeze-layers", type=str)
    parser.add_argument(
        "--action-padding-in-tj", action="store_true", default=None)
    parser.add_argument(
        "--append-objective-to-tj", action="store_true", default=None)
    parser.add_argument(
        "--walkthrough-guided-exploration", action="store_true",
        default=None, help="only effective during training")
    parser.add_argument(
        "--prob-complete-walkthrough", type=float, default=None,
        help="the probability of using all steps in walkthrough at training")
    parser.add_argument(
        "--num-tokens", type=int, help="maximum length for trajectory as input")
    return parser


def get_parser() -> ArgumentParser:
    """
    Get arg parser for different modules
    """

    parser = ArgumentParser(
        argument_default=None, parents=[hp_parser()],
        conflict_handler='resolve')
    parser.add_argument('--model-dir', type=str, required=True)

    subparsers = parser.add_subparsers(
        dest='mode',
        help="[train-dqn|eva--dqn|train-student|eval-student|gen-data]")

    teacher_parser = subparsers.add_parser('train-dqn')
    teacher_parser.add_argument(
        '--game-path', type=str, help='[a dir|a game file]', required=True)
    teacher_parser.add_argument('--f-games', type=str)
    teacher_parser.add_argument(
        '--request-obs-inv', action='store_true', default=False)

    eval_parser = subparsers.add_parser('eval-dqn')
    eval_parser.add_argument(
        '--eval-mode', type=str, default='eval',
        help='[eval|dev-eval|full-eval]')
    eval_parser.add_argument('--game-path', type=str, required=True)
    eval_parser.add_argument('--f-games', type=str)
    eval_parser.add_argument('--n-gpus', type=int, default=1)
    eval_parser.add_argument('--debug', action='store_true', default=False)
    eval_parser.add_argument('--load-best', action='store_true', default=False)
    eval_parser.add_argument('--restore-from', type=str)
    eval_parser.add_argument('--ckpt-range-min', type=int)
    eval_parser.add_argument('--ckpt-range-max', type=int)

    student_parser = subparsers.add_parser('train-student')
    student_parser.add_argument('--data-path', type=str, required=True)
    student_parser.add_argument('--learner-clazz', type=str)
    student_parser.add_argument('--n-epochs', type=int, default=1000)

    student_eval_parser = subparsers.add_parser('eval-student')
    student_eval_parser.add_argument('--data-path', type=str, required=True)
    student_eval_parser.add_argument('--learner-clazz', type=str)
    student_eval_parser.add_argument('--n-gpus', type=int, default=1)
    student_eval_parser.add_argument(
        '--debug', action='store_true', default=False)
    student_eval_parser.add_argument('--ckpt-range-min', type=int)
    student_eval_parser.add_argument('--ckpt-range-max', type=int)

    snn_gen_parser = subparsers.add_parser('gen-snn')
    snn_gen_parser.add_argument('--data-path', type=str, required=True)
    snn_gen_parser.add_argument('--learner-clazz', type=str)

    gen_data_parser = subparsers.add_parser('gen-data')
    gen_data_parser.add_argument('--game-path', type=str, required=True)
    gen_data_parser.add_argument('--f-games', type=str)
    gen_data_parser.add_argument(
        '--load-best', action='store_true', default=False)
    gen_data_parser.add_argument('--restore-from', type=str)
    gen_data_parser.add_argument('--epoch-size', type=int)
    gen_data_parser.add_argument('--epoch-limit', type=int, default=5)
    gen_data_parser.add_argument('--max-randomness', type=float, default=0.5)
    return parser


def run_agent(
        agent: BaseAgent, game_env: Env, nb_games: int, nb_epochs: int
) -> None:
    """
    Run a train agent on given games.

    Args:
        agent: an agent extends the base agent,
         see :py:class:`deepword.agents.base_agent.BaseAgent`.
        game_env: game Env, from gym
        nb_games: number of games
        nb_epochs: number of epochs for training
    """

    logger = logging.getLogger("train")
    for epoch_no in trange(nb_epochs):
        for game_no in trange(nb_games):
            logger.info("playing game epoch_no/game_no: {}/{}".format(
                epoch_no, game_no))
            try:
                obs, infos = game_env.reset()
                scores = [0] * len(obs)
                dones = [False] * len(obs)
                steps = [0] * len(obs)
                while not all(dones):
                    # Increase step counts.
                    steps = ([step + int(not done)
                              for step, done in zip(steps, dones)])
                    commands = agent.act(obs, scores, dones, infos)
                    obs, scores, dones, infos = game_env.step(commands)
                # Let the agent knows the game is done.
                agent.act(obs, scores, dones, infos)
            except Exception as e:
                eprint("training game {} error: {}".format(game_no, e))
                eprint("inform agent to finish current episode...")
                if agent._episode_has_started:
                    agent._episode_has_started = False
        if agent.total_t >= agent.hp.observation_t + agent.hp.annealing_eps_t:
            logger.info("training steps exceed MAX, stop training ...")
            logger.info("total training steps: {}".format(
                agent.total_t - agent.hp.observation_t))
            logger.info("save final model and snapshot ...")
            agent.save_snapshot()
            agent.core.save_model()
            return


def run_agent_v2(
        agent: BaseAgent, game_env: Env, nb_games: int, nb_epochs: int
) -> None:
    """
    Run a train agent on given games.
    Proactively request `look` and `inventory` results from games to substitute
    the description and inventory parts of infos.
    This is useful when games don't provide description and inventory, e.g. for
    Z-machine games.

    NB: This will incur extra steps for game playing, remember to use 3-times of
    previous step quota. E.g. previously use 100 max steps, now you need 100 * 3
    max steps.

    See :py:func:`deepword.main.run_agent`
    """

    logger = logging.getLogger("train")
    for epoch_no in trange(nb_epochs):
        for game_no in trange(nb_games):
            logger.info("playing game epoch_no/game_no: {}/{}".format(
                epoch_no, game_no))

            obs, infos = game_env.reset()
            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            # TODO: be cautious about the local variable problem
            look_res = [""] * len(obs)
            inventory_res = [""] * len(obs)
            while not all(dones):
                # TODO: get fake description from an extra look per step
                look_res, _, _, _ = game_env.step(["look"])
                infos['description'] = look_res
                inventory_res, _, _, _ = game_env.step(["inventory"])
                infos['inventory'] = inventory_res
                # Increase step counts.
                steps = ([step + int(not done)
                          for step, done in zip(steps, dones)])
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = game_env.step(commands)
            # Let the agent knows the game is done.
            # last state obs + inv copy previous state
            # TODO: this is OK for now, since we don't use last states for SNN
            infos['description'] = look_res
            infos['inventory'] = inventory_res
            agent.act(obs, scores, dones, infos)
        if agent.total_t >= agent.hp.observation_t + agent.hp.annealing_eps_t:
            logger.info("training steps exceed MAX, stop training ...")
            logger.info("total training steps: {}".format(
                agent.total_t - agent.hp.observation_t))
            logger.info("save final model and snapshot ...")
            agent.save_snapshot()
            agent.core.save_model()
            return


class AlfredDemangler(textworld.core.Wrapper):

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._game.infos)
        for info in self._game.infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


def train(
        hp: HParams, model_dir: str,
        game_dir: str, f_games: Optional[str] = None,
        func_run_agent: Callable[[BaseAgent, Env, int, int], None] = run_agent
) -> None:
    """
    train an agent

    Args:
        hp: hyper-parameters see :py:mod:`deepword.hparams`
        model_dir: model dir
        game_dir: game dir with ulx games
        f_games: game name to select from `game_dir`
        func_run_agent: how to run the agent and games,
         see :py:func:`deepword.main.run_agent`
    """

    logger = logging.getLogger('train')
    train_games = load_alfworld_games(game_dir)
    logger.info("load {} game files".format(len(train_games)))
    # nb epochs could only be an estimation since steps per episode is unknown
    nb_epochs = (hp.annealing_eps_t // len(train_games) // 10) + 1

    agent_clazz = agent_name2clazz(hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.train()

    requested_infos = agent.select_additional_infos()

    env_id = textworld.gym.register_games(
        train_games, requested_infos, batch_size=1,
        max_episode_steps=hp.game_episode_terminal_t, name="training",
        wrappers=[AlfredDemangler])
    env = gym.make(env_id)
    try:
        func_run_agent(agent, env, len(train_games), nb_epochs)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error("error: {}".format(e))
        traceback.print_exception(
            exc_type, exc_value, exc_traceback, limit=None, file=sys.stdout)
    env.close()


def train_v2(hp, model_dir, game_dir, f_games=None):
    """
    Train DQN agents by proactively requesting description and inventory

    max step per episode will be enlarged by 3-times.

    see :py:func:`deepword.main.train`
    """

    assert "game_episode_terminal_t" in hp, \
        "cannot find game_episode_terminal_t in hp"
    eprint("Requested game_episode_terminal_t: {}".format(
        hp.game_episode_terminal_t))
    hp.set_hparam('game_episode_terminal_t', hp.game_episode_terminal_t * 3)
    eprint("New game_episode_terminal_t: {}".format(hp.game_episode_terminal_t))

    # use run_agent_v2 to manually request description and inventory
    train(hp, model_dir, game_dir, f_games, func_run_agent=run_agent_v2)


def process_hp(args) -> HParams:
    """
    Load hyperparameters from three location
    1. config file in `model_dir`
    2. pre config files
    3. cmd line args

    Args:
        args: cmd line args

    Returns:
        hyperparameters
    """

    fn_hparams = os.path.join(args.model_dir, "hparams.json")
    if os.path.isfile(fn_hparams):
        model_config_file = fn_hparams
    else:
        model_config_file = None

    hp = load_hparams(
        fn_model_config=model_config_file, cmd_args=vars(args),
        fn_pre_config=args.config_file)
    return hp


warning_hparams_exist = """
hparams.json exists! Some hyper-parameter set from CMD and the pre-config file
will be disabled. Make sure to clear model_dir first if you want to train a
new agent from scratch!
""".replace("\n", " ").strip()


def process_train_dqn(args):
    """
    Train DQN models
    """

    fn_hparams = os.path.join(args.model_dir, "hparams.json")
    if os.path.isfile(fn_hparams):
        eprint(colored(warning_hparams_exist, "red", attrs=["bold"]))
    hp = process_hp(args)
    setup_train_log(args.model_dir)
    if args.request_obs_inv:
        train_v2(
            hp, args.model_dir, game_dir=args.game_path, f_games=args.f_games)
    else:
        train(
            hp, args.model_dir, game_dir=args.game_path, f_games=args.f_games)


def process_train_student(args):
    """
    Train student models
    """

    fn_hparams = os.path.join(args.model_dir, "hparams.json")
    if os.path.isfile(fn_hparams):
        eprint(colored(warning_hparams_exist, "red", attrs=["bold"]))
    hp = process_hp(args)
    setup_train_log(args.model_dir)
    learner_clazz = learner_name2clazz(hp.learner_clazz)
    learner = learner_clazz(hp, args.model_dir, args.data_path)
    learner.train(n_epochs=args.n_epochs)


def process_snn_input(args):
    """
    generate snn input
    """

    fn_hparams = os.path.join(args.model_dir, "hparams.json")
    if os.path.isfile(fn_hparams):
        eprint(colored(warning_hparams_exist, "red", attrs=["bold"]))
    hp = process_hp(args)
    setup_train_log(args.model_dir)
    learner_clazz = learner_name2clazz(hp.learner_clazz)
    learner = learner_clazz(hp, args.model_dir, args.data_path)
    learner.preprocess_input(data_dir=args.data_path)


def process_eval_student(args):
    """
    Evaluate student models
    """

    hp = process_hp(args)
    assert hp.learner_clazz == "SwagLearner" or \
           hp.learner_clazz == "SNNLearner" or \
           hp.learner_clazz == "NLULearner"
    learner_clazz = learner_name2clazz(hp.learner_clazz)

    setup_eval_log(log_filename=None)
    steps, step2ckpt = list_checkpoints(
        args.model_dir,
        range_min=args.ckpt_range_min, range_max=args.ckpt_range_max)
    eprint("evaluate {} checkpoints".format(len(steps)))

    for step in steps[::-1]:
        tester = learner_clazz(
            hp, args.model_dir, train_data_dir=None,
            eval_data_path=args.data_path)
        acc, total = tester.test(restore_from=step2ckpt[step])
        eprint("eval step: {}, acc: {}, total: {}".format(step, acc, total))


def process_eval_dqn(args):
    """
    Evaluate dqn models
    """

    hp = process_hp(args)
    setup_eval_log(log_filename=None)
    logger = logging.getLogger('eval-dqn')
    # eval_games = load_game_files(args.game_path, args.f_games)
    eval_games = load_alfworld_games(args.game_path)
    logger.info("load {} game files".format(len(eval_games)))

    if args.eval_mode == "eval":
        eval_player = MultiGPUsEvalPlayer(
            hp, args.model_dir, eval_games, args.n_gpus, args.load_best)
        eval_player.evaluate(
            restore_from=args.restore_from, debug=args.debug)
    elif args.eval_mode == "dev-eval":
        eval_player = LoopDogEvalPlayer()
        eval_player.start(
            hp, args.model_dir, eval_games, args.n_gpus, args.debug)
    elif args.eval_mode == "full-eval":
        eval_player = FullDirEvalPlayer()
        eval_player.start(
            hp, args.model_dir, eval_games, args.n_gpus, args.debug,
            range_min=args.ckpt_range_min, range_max=args.ckpt_range_max)
    else:
        raise ValueError()


def process_gen_data(args):
    """
    Generate training data from a teacher model
    """

    hp = process_hp(args)
    setup_eval_log(log_filename=None)

    game_files = load_game_files(args.game_path, args.f_games)

    eprint("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    eprint("games for data generation: \n{}".format(
        "\n".join(sorted(game_names))))

    # make sure epoch_size equals to replay_mem
    # if set, always use epoch_size;
    # otherwise, use replay_mem.
    # TODO: don't set replay_mem directly, because the replay_mem in cmd_args
    #   cannot reset replay_mem in hparams.json
    if args.epoch_size is None:
        args.epoch_size = hp.replay_mem
    else:
        hp.set_hparam("replay_mem", args.epoch_size)
    eprint("effective replay_mem: {}, epoch_size: {}".format(
        hp.replay_mem, args.epoch_size))

    # need to compute policy at every step
    hp.set_hparam("always_compute_policy", True)
    hp.set_hparam("max_snapshot_to_keep", args.epoch_limit)
    assert hp.agent_clazz == "TeacherAgent" or \
           hp.agent_clazz == "DSQNZorkAgent" or \
           hp.agent_clazz == "DSQNAgent", "Not supported agent class"

    agent_clazz = agent_name2clazz(hp.agent_clazz)
    agent = agent_clazz(hp, args.model_dir)
    agent.eval(load_best=args.load_best)

    if hp.agent_clazz == "DSQNZorkAgent":
        agent_collect_data_v2(
            agent, game_files, hp.game_episode_terminal_t,
            args.epoch_size, args.epoch_limit, args.max_randomness)
    else:
        agent_collect_data(
            agent, game_files, hp.game_episode_terminal_t,
            args.epoch_size, args.epoch_limit, args.max_randomness)


def main(args):
    with open(conventions.logo_file) as f:
        logo = "".join(f.readlines())
    eprint(logo)
    time.sleep(3)

    args.model_dir = args.model_dir.rstrip('/')
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    if args.mode == "train-dqn":
        process_train_dqn(args)
    elif args.mode == "train-student":
        process_train_student(args)
    elif args.mode == "eval-student":
        process_eval_student(args)
    elif args.mode == "gen-snn":
        process_snn_input(args)
    elif args.mode == "eval-dqn":
        process_eval_dqn(args)
    elif args.mode == "gen-data":
        process_gen_data(args)
    else:
        raise ValueError("please choose mode")


if __name__ == '__main__':
    main(get_parser().parse_args())
