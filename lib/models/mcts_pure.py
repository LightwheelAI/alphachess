import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """A random rollout policy function used during the rollout phase."""
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """Returns uniform probabilities and a score of 0 for the current state."""
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode:
    """A node in the MCTS tree, tracking its value Q, prior P, and visit count."""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # Map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand the node by creating new children."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select the action with the maximum value (Q + U)."""
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update the node's value based on leaf evaluation."""
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Recursively update the node and its ancestors."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate the node's value combining Q and U."""
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if the node is a leaf (no children)."""
        return not self._children

    def is_root(self):
        """Check if the node is the root node."""
        return self._parent is None


class MCTS:
    """Monte Carlo Tree Search implementation."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Initialize MCTS.
        
        :param policy_value_fn: Function to get action probabilities and score.
        :param c_puct: Exploration parameter controlling the balance between exploration and exploitation.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from root to leaf, and propagate the value back."""
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        end, winner = state.game_end()

        if not end:
            node.expand(action_probs)

        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Evaluate the value of the state using the rollout policy."""
        player = state.get_current_player_id()
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")

        return 0 if winner == -1 else (1 if winner == player else -1)

    def get_move(self, state):
        """Run playouts and return the most visited action."""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping known information about the subtree."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTS_Pure:
    """AI player based on MCTS."""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """Get the action for the current board state."""
        if board.availables:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return f"MCTS {self.player}"