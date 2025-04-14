import sys
import numpy as np
import random
import copy
import math
from collections import defaultdict
from tqdm import trange
import pickle
class MCTS_Node:
    def __init__(self,move_left, turn, parent=None, action=None,prior=0.0):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.turn = turn
        self.move_left = move_left
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.prior = prior
        # List of untried actions based on the current state's legal moves
        self.untried_action = []
        self.expand = False

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return self.expand
class MCTS:
    def __init__(self, approximator, iterations=200, simulation_batch=1, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.simulation_batch = simulation_batch

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        maxchild = None
        maxucb = -10000000000
        for c in node.children.values():
            ucb = (((-1 if node.turn != c.turn else 1)*c.total_reward / c.visits) if c.visits > 0 else 0) + self.c * c.prior * math.sqrt(node.visits+1) / (1 + c.visits)
            if ucb > maxucb:
                maxucb = ucb
                maxchild = c
        return maxchild


    def rollout(self, env,done,move_left,last_board,last_turn,last_value):
        sim_env = copy.deepcopy(env)
        depth = self.rollout_depth
        player = sim_env.turn
        threat = []
        last_pos = [[],[],[]]
        while done == 0 and depth > 0:
            if sim_env.check_instant_win(move_left,position=last_pos[sim_env.turn]): # instant win
                done = sim_env.turn
                break
            if len(threat) > move_left: # cannot defend
                done =  3 -sim_env.turn
                break
            position = threat
            for a in position: #defend against threats
                sim_env.board[a] = sim_env.turn
            move_left -= len(position)
            if move_left > 0:
                legalmove = np.argwhere(sim_env.board == 0)
                if len(legalmove) == 0:
                    break
                position += [tuple(a) for a in random.sample(list(legalmove),min(move_left,len(legalmove)))]
                for a in position:
                    sim_env.board[a] = sim_env.turn
            last_pos[sim_env.turn] = position
            done = sim_env.check_win(position=position)
            threat = sim_env.check_instant_win(2,position=last_pos[sim_env.turn])
            sim_env.turn = 3 - sim_env.turn
            depth -= 1
            move_left = 2
        #print(sim_env.board,file=sys.stderr)
        if done != 0:
            return 1e9 if done == player else -1e9
        if depth == 0: # cutoff
            #print(self.approximator.value(sim_env.board,player,last_board=last_board,last_turn=last_turn,last_value=last_value)-self.approximator.value(sim_env.board,player),file=sys.stderr)
            return self.approximator.value(sim_env.board,player,last_board=last_board,last_turn=last_turn,last_value=last_value)
        if done == 0:
            return 0
        

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            #print(node)
            #print(node.parent)
            node.visits += 1
            node.total_reward += reward
            if node.parent is not None and node.turn != node.parent.turn:
                reward = -reward 
            node = node.parent


    def run_simulation(self, root,env,last_board,last_turn,last_value):
        def compute_policy(board): # the shortest distance on board
            rows, cols = board.shape
            distances = np.full((rows, cols), 39) 
            for r in range(rows):
                for c in range(cols):
                    if board[r, c] != 0:  
                        distances[r, c] = 0
                    else:
                        if r > 0:
                            distances[r, c] = min(distances[r, c], distances[r - 1, c] + 1)
                        if c > 0:
                            distances[r, c] = min(distances[r, c], distances[r, c - 1] + 1)
            for r in range(rows - 1, -1, -1):
                for c in range(cols - 1, -1, -1):
                    if r < rows - 1:
                        distances[r, c] = min(distances[r, c], distances[r + 1, c] + 1)
                    if c < cols - 1:
                        distances[r, c] = min(distances[r, c], distances[r, c + 1] + 1)
            return distances
            
        def label_to_index(col_char):
            """Converts letter to column index (accounting for missing 'I')."""
            col_char = col_char.upper()
            if col_char >= 'J':  # 'I' is skipped
                return ord(col_char) - ord('A') - 1
            else:
                return ord(col_char) - ord('A')
        def action_to_pos(stone):
            return int(stone[1:]) - 1, label_to_index(stone[0])
        def softmax(x):
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x)

        sim_env = copy.deepcopy(env)
        node = root
        done = 0
        depth = 0
        while done == 0 and node.fully_expanded():
            depth += 1
            node = self.select_child(node)
            state,done = sim_env.step(node.action)
            sim_env.turn = node.turn
        if done == 0 and not node.fully_expanded():
            if not node.expand:
                node.untried_action = sim_env.get_legal_moves()
                node.expand = True
                policy = compute_policy(sim_env.board) # get distance to the closest stone
                policy = softmax([-policy[action_to_pos(action)] for action in node.untried_action])

                for i,action in enumerate(node.untried_action): # generate all children with priors
                    if node.move_left == 1:
                        child =  MCTS_Node(2, 3 - node.turn, parent=node, action=action,prior=policy[i])
                    else:
                        child =  MCTS_Node(1, node.turn, parent=node, action=action,prior=policy[i])
                    node.children[action] = child
            #print(node.children,file=sys.stderr)
            for _ in range(self.simulation_batch):
                sim_env2 = copy.deepcopy(sim_env)
                action = self.select_child(node).action # select a child to expand
                if action in node.untried_action:
                    node.untried_action.remove(action)
                child = node.children[action]
                state , done = sim_env2.step(action)
                sim_env2.turn = child.turn
                rollout_reward = self.rollout(sim_env2,done,child.move_left,last_board,last_turn,last_value)
                self.backpropagate(child, rollout_reward)
        else:
            rollout_reward = 1e9 if done == node.turn else -1e9
            if done == 0:
                rollout_reward = 0
            self.backpropagate(node, rollout_reward)
        return depth