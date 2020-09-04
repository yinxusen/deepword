import unittest

from bitarray import bitarray

from deepword.agents.utils import *
from deepword.utils import bytes2idx


class TestTeacherData(unittest.TestCase):
    @classmethod
    def gen_random_bit_array(cls, size: int):
        n_samples = np.random.randint(0, size // 2)
        np_mask = np.sort(np.random.choice(
            np.arange(size), size=n_samples, replace=False))
        bit_mask_vec = bitarray(size + 1, endian="little")
        bit_mask_vec[::] = False
        bit_mask_vec[-1] = True  # to avoid tail trimming for bytes
        for k in np_mask:
            bit_mask_vec[k] = True
        return bit_mask_vec.tobytes(), np_mask

    def test_bit_mask(self):
        n_samples = 1000
        for _ in range(n_samples):
            n_actions = np.random.randint(1, 1024)
            bit_mask, np_masks = self.gen_random_bit_array(n_actions)
            converted_np_mask = bytes2idx(bit_mask, size=n_actions + 1)
            self.assertEqual(len(np_masks), len(converted_np_mask))
            self.assertTrue(np.all(np.equal(np_masks, converted_np_mask)))
