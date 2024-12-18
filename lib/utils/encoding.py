import numpy as np
from collections import deque
import random
from lib.config import cfg

# Mapping from numbers to arrays
num2array = {
    1: np.array([1, 0, 0, 0, 0, 0, 0]),
    2: np.array([0, 1, 0, 0, 0, 0, 0]),
    3: np.array([0, 0, 1, 0, 0, 0, 0]),
    4: np.array([0, 0, 0, 1, 0, 0, 0]),
    5: np.array([0, 0, 0, 0, 1, 0, 0]),
    6: np.array([0, 0, 0, 0, 0, 1, 0]),
    7: np.array([0, 0, 0, 0, 0, 0, 1]),
    8: np.array([-1, 0, 0, 0, 0, 0, 0]),
    9: np.array([0, -1, 0, 0, 0, 0, 0]),
    10: np.array([0, 0, -1, 0, 0, 0, 0]),
    11: np.array([0, 0, 0, -1, 0, 0, 0]),
    12: np.array([0, 0, 0, 0, -1, 0, 0]),
    13: np.array([0, 0, 0, 0, 0, -1, 0]),
    14: np.array([0, 0, 0, 0, 0, 0, -1]),
    15: np.array([0, 0, 0, 0, 0, 0, 0])
}

def array2num(array):
    """Convert an array back to its corresponding number."""
    for num, num_array in num2array.items():
        if (num_array == array).all():
            return num
    raise ValueError("Array not found in mapping.")

def state_list2state_num_array(state_list):
    """Convert a list of states to a state number array."""
    state_array = np.zeros((10, 9, 7))
    for i in range(10):
        for j in range(9):
            state_array[i][j] = num2array[state_list[i][j]]
    return state_array

def zip_state_mcts_prob(data_tuple):
    """Zip the state, MCTS probability, and winner into a compact form."""
    state, mcts_prob, winner = data_tuple
    state = state.reshape((9, -1))
    mcts_prob = mcts_prob.reshape((2, -1))
    state = zip_array(state)
    mcts_prob = zip_array(mcts_prob)
    return state, mcts_prob, winner

def recovery_state_mcts_prob(data_tuple):
    """Recover the state and MCTS probability from a compact form."""
    state, mcts_prob, winner = data_tuple
    state = recovery_array(state)
    mcts_prob = recovery_array(mcts_prob)
    state = state.reshape((9, 10, 9))
    mcts_prob = mcts_prob.reshape(2086)
    return state, mcts_prob, winner

def zip_array(array, data=0.):
    """Compress an array into a sparse representation."""
    zip_res = [[len(array), len(array[0])]]
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] != data:
                zip_res.append([i, j, array[i][j]])
    return np.array(zip_res, dtype=object)

def recovery_array(array, data=0.):
    """Recover a sparse array back to its original form."""
    recovery_res = [[data] * array[0][1] for _ in range(array[0][0])]
    for i in range(1, len(array)):
        recovery_res[array[i][0]][array[i][1]] = array[i][2]
    return np.array(recovery_res)