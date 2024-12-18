"""Monte Carlo Tree Search (MCTS)"""

import numpy as np
import copy
from lib.config import cfg


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class TreeNode:
    """
    TreeNode represents a node in the MCTS tree. 
    It records the action taken to reach this node, its Q value, prior probability P, and visit counts.
    """

    def __init__(self, parent, prior_p):
        """
        Initialize a TreeNode.
        
        :param parent: Parent node
        :param prior_p: Prior probability for this node
        """
        self._parent = parent
        self._children = {}  # Action to TreeNode mapping
        self._n_visits = 0    # Visit count
        self._Q = 0           # Average action value
        self._u = 0           # Upper confidence bound (PUCT)
        self._P = prior_p

    def expand(self, action_priors):
        """Expand the node by creating new child TreeNodes."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select the child node with the highest value (Q + U)."""
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """Calculate the value of this node."""
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """Update the node value based on leaf evaluation."""
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Recursively update this node and its ancestors."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if the node is a leaf (no children)."""
        return not self._children

    def is_root(self):
        """Check if this node is the root node."""
        return self._parent is None


class MCTS:
    """Monte Carlo Tree Search algorithm."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """
        Initialize MCTS.

        :param policy_value_fn: Function to get action probabilities and board evaluation
        :param c_puct: Exploration parameter
        :param n_playout: Number of playouts per move
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Perform a playout and update the tree nodes based on evaluations."""
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        
        if not end:
            node.expand(action_probs)
        else:
            leaf_value = 0.0 if winner == -1 else (1.0 if winner == state.get_current_player_id() else -1.0)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Get action probabilities from the root node."""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """Update the root node based on the last move played."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'


class MCTSPlayer:
    """AI player based on MCTS."""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        """Reset the search tree for the player."""
        self.mcts.update_with_move(-1)

    def __str__(self):
        return f'MCTS {self.player}'

    def get_action(self, board, temp=1e-3, return_prob=0):
        """Get the action to be taken based on the current board state."""
        move_probs = np.zeros(2086)

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(cfg.model.dirichlet * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        return (move, move_probs) if return_prob else move