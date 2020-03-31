import unittest
import random

from deeptextworld.agents.utils import *
from deeptextworld.action import ActionCollector


def gen_rand_str(vocab: List[str], length_up_to: int, n_rows: int) -> List[str]:
    res = []
    for i in range(n_rows):
        n_cols = random.randint(0, length_up_to)
        res.append(" ".join(random.choices(vocab, k=n_cols)))
    return res


class TestAgentInput(unittest.TestCase):
    def test_drrn_action_input(self):
        tok = NLTKTokenizer(
            vocab_file=conventions.nltk_vocab_file, do_lower_case=True)
        ac = ActionCollector(
            tok, n_tokens=10,
            unk_val_id=tok.vocab[conventions.nltk_unk_token],
            padding_val_id=tok.vocab[conventions.nltk_padding_token])

        game_ids = gen_rand_str(
            list(tok.vocab.keys()), length_up_to=10, n_rows=1000)

        for gid in game_ids:
            ac.add_new_episode(gid)
            ac.extend(gen_rand_str(
                list(tok.vocab.keys()), length_up_to=10,
                n_rows=random.randint(1, 100)))

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


if __name__ == '__main__':
    unittest.main()
