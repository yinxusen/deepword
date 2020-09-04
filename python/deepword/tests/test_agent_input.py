import random
import unittest

from tensorflow.contrib.training import HParams

from deepword.action import ActionCollector
from deepword.agents.base_agent import BaseAgent
from deepword.agents.utils import *
from deepword.hparams import copy_hparams

tokenizer_hp = HParams(
    vocab_size=0,
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
    tokenizer_type=None,
    use_glove_emb=False)


def gen_rand_str(
        vocab: List[str], length_up_to: int, n_rows: int,
        allow_empty_str: bool = True) -> List[str]:
    res = []
    assert length_up_to > 0
    for i in range(n_rows):
        n_cols = random.randint(0 if allow_empty_str else 1, length_up_to)
        res.append(" ".join(random.choices(vocab, k=n_cols)))
    return res


def gen_action_master(
        vocab: List[str], turns_up_to: int, n_rows: int
) -> List[List[ActionMaster]]:
    assert turns_up_to > 0, "at least need 1-turn of action-master"
    res = []
    for i in range(n_rows):
        n_turns = np.random.randint(1, turns_up_to + 1)
        rnd_actions = gen_rand_str(vocab, 10, n_turns, allow_empty_str=False)
        rnd_masters = gen_rand_str(vocab, 50, n_turns, allow_empty_str=False)
        res.append(
            [ActionMaster(a, m) for a, m in zip(rnd_actions, rnd_masters)])
    return res


def gen_action_collector(hp: HParams, tokenizer: Tokenizer) -> ActionCollector:
    ac = ActionCollector(
        tokenizer, n_tokens=10,
        unk_val_id=hp.unk_val_id, padding_val_id=hp.padding_val_id)
    vocab = list(tokenizer.vocab.keys())

    game_ids = gen_rand_str(vocab, length_up_to=10, n_rows=1000)

    for gid in game_ids:
        ac.add_new_episode(gid)
        ac.extend(gen_rand_str(
            vocab, length_up_to=10, n_rows=random.randint(1, 100)))

    return ac


class TestAgentInput(unittest.TestCase):
    def test_dqn_input(self):
        hp = copy_hparams(tokenizer_hp)
        hp.set_hparam("tokenizer_type", "Bert")
        hp, tokenizer = BaseAgent.init_tokens(hp)
        vocab = list(tokenizer.vocab.keys())

        for _ in range(1000):
            num_tokens = np.random.randint(1, 1024)
            trajectories = gen_action_master(vocab, turns_up_to=5, n_rows=8)
            src, src_len, src_master_mask = batch_dqn_input(
                trajectories, tokenizer, num_tokens, hp.padding_val_id,
                with_action_padding=False, max_action_size=None)
            self.assertTrue(all([0 < x <= num_tokens for x in src_len]))
            # TODO: how to test generated src?

        # test with action padding
        for _ in range(1000):
            num_tokens = 256
            trajectories = gen_action_master(vocab, turns_up_to=5, n_rows=8)
            max_action_size = np.random.randint(1, num_tokens)
            src, src_len, src_master_mask = batch_dqn_input(
                trajectories, tokenizer, num_tokens, hp.padding_val_id,
                with_action_padding=True, max_action_size=max_action_size)
            self.assertTrue(all([0 < x <= num_tokens for x in src_len]))
            for ss, ll, mm in zip(src, src_len, src_master_mask):
                self.assertTrue(all([x == 0 for x in ss[ll:]]))
                if ll < num_tokens:  # for those untrimmed trajectories
                    action_token_ids = np.where(np.asarray(mm[:ll]) == 0)[0]
                    next_starter_minimum = 0
                    for idx in action_token_ids:
                        if idx >= next_starter_minimum:
                            self.assertTrue(
                                idx + max_action_size - 1 in action_token_ids)
                            next_starter_minimum = idx + max_action_size
                        else:
                            self.assertTrue(idx - 1 in action_token_ids)
                else:  # for trimmed trajectories
                    pass

    def test_drrn_action_input(self):
        hp = copy_hparams(tokenizer_hp)
        hp.set_hparam("tokenizer_type", "NLTK")
        hp, tokenizer = BaseAgent.init_tokens(hp)
        ac = gen_action_collector(hp, tokenizer)
        game_ids = ac.get_game_ids()
        action_len = [ac.get_action_len(gid) for gid in game_ids]
        action_matrix = [ac.get_action_matrix(gid) for gid in game_ids]
        action_mask = [
            np.random.choice(
                list(range(len(actions))),
                size=random.randint(1, len(actions)), replace=False)
            for actions in action_matrix]
        expected_ids = [np.random.choice(idx, size=None) for idx in action_mask]

        (inp_matrix, inp_len, actions_repeats, id_real2mask
         ) = batch_drrn_action_input(action_matrix, action_len, action_mask)

        batch_ids = id_real2batch(expected_ids, id_real2mask, actions_repeats)

        for i in range(len(game_ids)):
            action1 = action_matrix[i][expected_ids[i]]
            action2 = inp_matrix[batch_ids[i]]
            self.assertTrue(np.all(np.equal(action1, action2)))

    def test_bert_commonsense_input(self):
        hp = copy_hparams(tokenizer_hp)
        hp.set_hparam("tokenizer_type", "Bert")
        hp, tokenizer = BaseAgent.init_tokens(hp)
        vocab = list(tokenizer.vocab.keys())
        ac = gen_action_collector(hp, tokenizer)
        game_ids = ac.get_game_ids()

        num_tokens = 256
        n_tokens_per_action = 10
        n_special_tokens = 3
        n_tj_tokens = num_tokens - n_tokens_per_action - n_special_tokens
        trajectories = gen_action_master(vocab, turns_up_to=5, n_rows=8)
        src, src_len, src_master_mask = batch_dqn_input(
            trajectories, tokenizer, n_tj_tokens, hp.padding_val_id)
        for s, l, m in zip(src, src_len, src_master_mask):
            gid = np.random.choice(game_ids)
            action_matrix = ac.get_action_matrix(gid)
            n_actions = len(action_matrix)
            action_mask = np.random.choice(
                np.arange(n_actions), size=np.random.randint(1, n_actions))
            inp, seg_tj_action, inp_size = bert_commonsense_input(
                action_matrix=action_matrix[action_mask],
                action_len=ac.get_action_len(gid)[action_mask],
                trajectory=s, trajectory_len=l,
                sep_val_id=hp.sep_val_id, cls_val_id=hp.cls_val_id,
                num_tokens=num_tokens)
            self.assertEqual(len(action_mask), len(inp))
            self.assertEqual(len(inp), len(seg_tj_action))
            self.assertEqual(len(inp), len(inp_size))
            self.assertTrue(all([0 < x <= num_tokens for x in inp_size]))
            for ii, ss, ll in zip(inp, seg_tj_action, inp_size):
                self.assertEqual(len(ii), num_tokens)
                self.assertEqual(ii[0], hp.cls_val_id)
                self.assertEqual(ii[ll-1], hp.sep_val_id)
                self.assertTrue(all([x == 0 for x in ii[ll:]]))
                self.assertTrue(hp.sep_val_id in ii[:ll-1])
                inner_sep_idx = list(ii[:ll-1]).index(hp.sep_val_id)
                self.assertTrue(all([x == 0 for x in ss[:inner_sep_idx+1]]))
                self.assertTrue(all([x == 1 for x in ss[inner_sep_idx+1:]]))

    def test_sample_ids(self):
        actions_repeats = np.random.randint(
            2, 1024, size=np.random.randint(1, 100))
        q_actions = np.random.random(np.sum(actions_repeats))
        selected_frequencies = np.zeros_like(q_actions)
        n_sampling_times = 10000
        for _ in range(n_sampling_times):
            k = np.random.randint(1, np.max(actions_repeats))
            action_ids = sample_batch_ids(q_actions, list(actions_repeats), k)
            self.assertEqual(len(action_ids), k * len(actions_repeats))
            sampled_q_actions = q_actions[action_ids].reshape([-1, k])
            self.assertTrue(
                [x == 0 for x in np.argmax(sampled_q_actions, axis=-1)])
            idx_starter = 0
            for n_repeats, ids in zip(
                    actions_repeats, np.asarray(action_ids).reshape([-1, k])):
                self.assertTrue(
                    [idx_starter <= x < idx_starter + n_repeats for x in ids])
                self.assertAlmostEqual(
                    q_actions[ids[0]],
                    np.max(q_actions[idx_starter: idx_starter + n_repeats]))
                if n_repeats >= k:  # assert selected items are unique
                    self.assertEqual(len(ids), len(set(ids)))
                selected_frequencies[ids] += 1
                idx_starter += n_repeats

        idx_starter = 0
        for n_repeats in actions_repeats:
            curr_frequencies = (
                selected_frequencies[idx_starter: idx_starter + n_repeats]
                / n_sampling_times)
            curr_max_id = np.argmax(curr_frequencies)
            self.assertAlmostEqual(curr_frequencies[curr_max_id], 1.0)
            others = np.concatenate(
                [curr_frequencies[:curr_max_id],
                 curr_frequencies[curr_max_id+1:]])
            # assert uniformly selected
            self.assertTrue([x < 0.1 for x in others - np.mean(others)])
            idx_starter += n_repeats

    def test_align_batch_str(self):
        hp = copy_hparams(tokenizer_hp)
        hp.set_hparam("tokenizer_type", "NLTK")
        hp, tokenizer = BaseAgent.init_tokens(hp)
        vocab = list(tokenizer.vocab.keys())

        for _ in range(100):
            num_tokens = np.random.randint(1, 1024)
            n_rows = 100
            data = gen_rand_str(
                vocab, length_up_to=num_tokens, n_rows=n_rows,
                allow_empty_str=True)
            ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
                   for s in data]
            str_len_allowance = 1000
            valid_len = [len(x) for x in ids]
            aligned_str, aligned_len = align_batch_str(
                ids, str_len_allowance, hp.padding_val_id, valid_len)

            self.assertTrue(np.all(aligned_len <= str_len_allowance))
            self.assertTrue(aligned_str.shape[0] == n_rows)
            self.assertTrue(aligned_str.shape[1] <= str_len_allowance)
            self.assertTrue(aligned_str.shape[1] <= max(valid_len))

            for s, l, raw_str, raw_len in zip(
                    aligned_str, aligned_len, ids, valid_len):
                self.assertTrue(l == min(raw_len, str_len_allowance))
                for i in range(min(l, raw_len)):
                    self.assertEqual(s[i], raw_str[i])
                for i in range(l, aligned_str.shape[1]):
                    self.assertEqual(s[i], hp.padding_val_id)


if __name__ == '__main__':
    unittest.main()
