import sys
import numpy as np
import random
import copy
import math
from collections import defaultdict
from tqdm import trange
import pickle

class MCTS_Node:
    def __init__(self,move_left, turn, parent=None, action=None):
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
        # List of untried actions based on the current state's legal moves
        self.untried_action = []
        self.expand = False

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return (len(self.untried_action) == 0 ) if self.expand else False
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
            ucb = (((-1 if node.turn != c.turn else 1)*c.total_reward / c.visits) if c.visits > 0 else 0) + self.c * math.sqrt(math.log(node.visits) / (c.visits))
            if ucb > maxucb:
                maxucb = ucb
                maxchild = c
        return maxchild


    def rollout(self, env,done,last_board,last_turn,last_value):
        sim_env = copy.deepcopy(env)
        depth = self.rollout_depth
        player = sim_env.turn
        while done == 0 and depth > 0:
            legalmove = np.argwhere(sim_env.board == 0)
            if len(legalmove) == 0:
                break
            position = [tuple(a) for a in random.sample(list(legalmove),min(2,len(legalmove)))]
            for a in position:
                sim_env.board[a] = sim_env.turn
            done = sim_env.check_win(position=position)
            sim_env.turn = 3 - sim_env.turn
            depth -= 1
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
        sim_env = copy.deepcopy(env)
        node = root
        done = 0
        depth = 0
        while done == 0 and node.fully_expanded():
            depth += 1
            node = self.select_child(node)
            state,done = sim_env.step(node.action)
            sim_env.turn = node.turn
        # TODO: Expansion: if the node has untried actions, expand one.
        if done == 0 and not node.fully_expanded():
            if not node.expand:
                node.untried_action = sim_env.get_legal_moves()
                node.expand = True
            action = random.choice(node.untried_action)
            node.untried_action.remove(action)
            state , done = sim_env.step(action)
            if node.move_left == 1:
                child =  MCTS_Node(2, 3 - node.turn, parent=node, action=action)
            else:
                child =  MCTS_Node(1, node.turn, parent=node, action=action)
            node.children[action] = child
            sim_env.turn = child.turn
            rollout_reward = self.rollout(sim_env,done,last_board,last_turn,last_value)
            self.backpropagate(child, rollout_reward)
        else:
            rollout_reward = 1e9 if done == node.turn else -1e9
            self.backpropagate(node, rollout_reward)
        return depth
