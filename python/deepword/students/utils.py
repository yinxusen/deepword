from collections import namedtuple
from typing import List, Tuple, Optional, Iterator

import numpy as np

from deepword.tokenizers import Tokenizer


class ActionMasterStr(namedtuple("ActionMaster", ("action", "master"))):
    pass


def dqn_input(
        trajectory: List[ActionMasterStr],
        tokenizer: Tokenizer,
        num_tokens: int,
        padding_val_id: int,
        with_action_padding: bool = False,
        max_action_size: Optional[int] = None
) -> Tuple[List[int], int, List[int]]:
    """
    Given a trajectory (a list of ActionMaster), get trajectory indexes, length
    and master mask (master marked as 1 while action marked as 0).
    Pad the trajectory to num_tokens.
    Pad actions if required.
    """
    trajectory_ids, raw_master_mask = tj2ids(
        trajectory, tokenizer,
        with_action_padding, max_action_size, padding_val_id)
    padding_size = num_tokens - len(trajectory_ids)
    if padding_size >= 0:
        src = trajectory_ids + [padding_val_id] * padding_size
        master_mask = raw_master_mask + [0] * padding_size
        src_len = len(trajectory_ids)
    else:
        src = trajectory_ids[-padding_size:]
        master_mask = raw_master_mask[-padding_size:]
        src_len = num_tokens
    return src, src_len, master_mask


def batch_dqn_input(
        trajectories: List[List[ActionMasterStr]],
        tokenizer: Tokenizer,
        num_tokens: int,
        padding_val_id: int,
        with_action_padding: bool = False,
        max_action_size: Optional[int] = None
) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    batch_src = []
    batch_src_len = []
    batch_mask = []
    for tj in trajectories:
        src, src_len, master_mask = dqn_input(
            tj, tokenizer, num_tokens, padding_val_id,
            with_action_padding, max_action_size)
        batch_src.append(src)
        batch_src_len.append(src_len)
        batch_mask.append(master_mask)
    return batch_src, batch_src_len, batch_mask


def align_batch_str(
        ids: List[List[int]], str_len_allowance: int,
        padding_val_id: int, valid_len: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a batch of string indexes.
    The maximum length will not exceed str_len_allowance.
    Each array of ids will be either padded or trimmed to reach
    the maximum length, notice that padded length won't be counted in as valid
    length.

    Args:
        ids: a list of array of index (int)
        str_len_allowance:
        padding_val_id:
        valid_len:

    Returns:
        aligned ids and aligned length
    """
    def align() -> Iterator[Tuple[List[int], int]]:
        m = min(str_len_allowance, np.max(valid_len))
        for s, l in zip(ids, valid_len):
            if 0 <= l < m:
                yield s[:l] + [padding_val_id] * (m - l), l
            else:
                yield s[:m], m

    aligned_ids, aligned_len = zip(*align())
    return np.asarray(aligned_ids), np.asarray(aligned_len)


def tj2ids(
        trajectory: List[ActionMasterStr],
        tokenizer: Tokenizer,
        with_action_padding: bool = False,
        max_action_size: Optional[int] = None,
        padding_val_id: int = 0) -> Tuple[List[int], List[int]]:
    """
    Convert a trajectory (list of ActionMaster) into ids
    Compute segmentation ids for masters (1) and actions (0)
    pad actions if required.
    """
    ids = []
    master_mask = []
    for am in trajectory:
        action_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(am.action))
        if with_action_padding:
            if len(action_ids) < max_action_size:
                action_ids += [padding_val_id] * (
                        max_action_size - len(action_ids))
            else:
                action_ids = action_ids[:max_action_size]
        master_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(am.master))
        ids += action_ids
        ids += master_ids
        master_mask += [0] * len(action_ids)
        master_mask += [1] * len(master_ids)
    return ids, master_mask
