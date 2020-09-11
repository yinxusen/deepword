from typing import Tuple, Optional, List

import numpy as np
from scipy import stats


def mean_confidence_interval(
        data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Given data, 1D np array, compute the mean and confidence intervals given
    confidence level.

    Args:
        data: 1D np array
        confidence: confidence level

    Returns:
        mean and confidence interval
    """

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = float(np.mean(a)), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


class UCBComputer(object):
    """
    Compute the Upper Confidence Bound actions during game playing at inference
    time only, when hidden states are fixed.

    The Abbasi-Yadkori, Pal, and Szepesvari bound for LinUCB.
        Cite: Improved Algorithms for Linear Stochastic Bandits,
         (Abbasi-Yadkori, Pal, and Szepesvari, 2011)
        See also: Learn What Not to Learn, (Tom Zahavy et al., 2019)

    We use the APS bound.
    """

    def __init__(self, d_states: int, d_actions: int):
        """
        V: covariance matrix for each action a
        lam: lambda to control parameter size in ridge regression
        r: R-sub-Gaussian
        s: bound for |theta_a|_2
        delta: with probability of 1 - delta, we have the bound

        Args:
            d_states: dimension of hidden states
            d_actions: number of actions
        """

        self.V: Optional[List[np.ndarray]] = None
        self.log_det_V: Optional[List[np.ndarray]] = None
        self.inv_V: Optional[List[np.ndarray]] = None
        self.lam = 0.5
        self.d_states = d_states
        self.d_actions = d_actions
        self.r = 1
        self.s = 1
        self.delta = 0.1 / self.d_actions
        self.log_lam = np.log(self.lam)
        self.log_delta = np.log(self.delta)

    def reset(self) -> None:
        """
        Reset to accept new episodes
        """

        self.V = [
            self.lam * np.eye(self.d_states) for _ in range(self.d_actions)]
        self.log_det_V = [np.log(np.linalg.det(a_mat)) for a_mat in self.V]
        self.inv_V = [np.linalg.inv(a_mat) for a_mat in self.V]

    def collect_sample(self, action_idx: int, h_state: np.ndarray) -> None:
        """
        Collect state-action pairs

        Args:
            action_idx: action index
            h_state: hidden state vector
        """

        self.V[action_idx] += np.outer(h_state, h_state)
        self.log_det_V[action_idx] = np.log(np.linalg.det(self.V[action_idx]))
        self.inv_V[action_idx] = np.linalg.inv(self.V[action_idx])

    def aps_bound(self, q_actions: np.ndarray, h_state: np.ndarray) -> float:
        """
        Compute APS bound

        Args:
            q_actions: Q-vector of actions
            h_state: hidden state of a game state

        Returns
            upper confidence bound of q_actions.
        """

        f = (0.5 * np.asarray(self.log_det_V) -
             0.5 * self.d_states * self.log_lam - self.log_delta)
        coeff = self.r * np.sqrt(2 * f) + np.sqrt(self.lam) * self.s
        bound = coeff * np.squeeze(np.asarray([
            np.sqrt(np.matmul(np.matmul(np.transpose(h_state), inv_v), h_state))
            for inv_v in self.inv_V]))
        normed_bound = bound / (np.max(bound) + 0.01)
        ucb_q_actions = q_actions + 0.2 * normed_bound
        return ucb_q_actions
