# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True
    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)
import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from tqdm import trange

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
def step_half(env,action):
    """Execute one action, returns all possible board configuration. (board, p , done)"""
    self = copy.deepcopy(env)
    if action == 0:
        moved = self.move_up()
    elif action == 1:
        moved = self.move_down()
    elif action == 2:
        moved = self.move_left()
    elif action == 3:
        moved = self.move_right()
    else:
        moved = False
    self.last_move_valid = moved  # Record if the move was valid
    return self.board,self.score
    
class TD_MCTS_Node:
    def __init__(self, score,env=None,state=None, parent=None, action=None, chance=True,value=0):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0.0 # no division by zero
        self.total_reward = 0.0
        self.chance = chance # whether it is chance node
        self.expanded = False 
        self.value = value # value estimate
        self.untried_actions = []
        if self.chance:
            self.state = state
        else:
            self.state = env.board.copy()
            self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
            

    def fully_expanded(self):
        return self.expanded


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=1):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.normalizer = 30000 # normalize approximator values

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node,start_score):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            return node.children[action]
        maxchild = None
        maxucb = -10000
        for c in node.children.values():
            ucb = (c.total_reward / c.visits - start_score) / self.normalizer + self.c * math.sqrt(math.log(node.visits) / c.visits)
            if ucb > maxucb:
                maxucb = ucb
                maxchild = c
        return maxchild

    def rollout(self, sim_env, depth):
        sim_env = copy.deepcopy(sim_env)
        done = False
        while depth > 0 and not done:
          legalmove = [a for a in range(4) if sim_env.is_move_legal(a)]
          if not legalmove:
            break
          state, reward, done, _ = sim_env.step(random.choice(legalmove))
          depth -= 1
        return sim_env.score + self.approximator.value(sim_env.board)

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))
    def board_hash(self,board):
        return sum([self.tile_to_index(board[i//4][i%4]) << (i*5) for i in range(16)])
    def run_simulation(self, root,state):
        depth = 0
        node = root
        sim_env = self.create_env_from_state(state,node.score)
        start_score = node.score
        done = False
        while node.expanded:
            node = self.select_child(node,start_score) #step to chance node
            if not node.expanded:
                break
            state, score, done, _ = sim_env.step(node.action) # sample 1 afterstate
            idx = self.board_hash(state)
            if idx not in node.children: # extend if not visited
                node.children[idx] = TD_MCTS_Node(score,env=sim_env, parent=node, action=idx,chance=False)
            node = node.children[idx] # non chance node
            depth += 1
        if not done: # not terminal, expand
            node.expanded = True
            if node.chance: # chance node
                self.backpropagate(node,node.value + node.score) # propogate with already calculated value
            else: # non chance node
                rollout_reward = 0
                for action in [a for a in range(4) if sim_env.is_move_legal(a)]:
                    state,reward = step_half(sim_env,action) # without adding random tiles
                    node.children[action] = TD_MCTS_Node(reward,state=state, parent=node, action=action,chance=True,value=self.approximator.value(state)) # create chance node
                    rollout_reward = max(rollout_reward,node.children[action].value + reward)
                self.backpropagate(node,rollout_reward)
        else:
            self.backpropagate(node, sim_env.score)
        return depth
        # Backpropagate the obtained reward.
        

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        best_reward = 0
        best_action = 0
        for action, child in root.children.items():
            if  child.total_reward / child.visits > best_reward:
                best_reward = child.total_reward / child.visits
                best_action = action
        return best_action


def rot(coord,k):
  for i in range(k):
    coord = (coord[1], 3 - coord[0])
  return coord
def rot_pattern(pattern,k):
  return [rot(p,k) for p in pattern]
def flip(pattern):
  return [ (p[0],3-p[1]) for p in pattern ]


def check_game_over(board):
  if np.any(board == 0):
        return False
  for i in range(4):
      for j in range(3):
          if board[i, j] == board[i, j+1]:
              return False
  for j in range(4):
      for i in range(3):
          if board[i, j] == board[i+1, j]:
              return False
  return True

class NTupleApproximator:
    def __init__(self, board_size):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        patterns = [[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)],[(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],[(1,0),(1,1),(1,2),(1,3),(2,0),(2,1)]]
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        self.symetric_patterns = []
        for p in patterns:
          self.symetric_patterns += self.generate_symmetries(p)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        return [rot_pattern(pattern,k) for k in range(4)] + [flip(rot_pattern(pattern,k)) for k in range(4)]
    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        feature = []
        for c in coords:
          i,j = c
          feature.append(self.tile_to_index(board[i,j]))
        return tuple(feature)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        if check_game_over(board):
          return 0
        value = 0
        idx = 0
        for pattern in self.symetric_patterns:
            f = self.get_feature(board, pattern)
            if f not in self.weights[idx//8]:
                self.weights[idx//8][f] = 0
            value += self.weights[idx//8][f]
            idx += 1
        return value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        idx = 0
        N = len(self.symetric_patterns)
        for pattern in self.symetric_patterns:
            f = self.get_feature(board, pattern)
            self.weights[idx//8][f] += alpha * delta / N
            idx += 1


import requests
approximator = NTupleApproximator(board_size=4)
def download_public_file(file_url, destination_path):
    response = requests.get(file_url)
    response.raise_for_status()  # Ensure the request was successful
    
    with open(destination_path, 'wb') as file:
        file.write(response.content)
    #print(f"File downloaded successfully to {destination_path}")
import os
if not os.path.isfile('approximator.weights'):
    download_public_file('https://drive.usercontent.google.com/download?id=1YzHJwp5Kx8F0ueHXOALY44Zz4KCRclcr&export=download&authuser=0&confirm=t&uuid=7b9a2567-1de1-491b-b319-3ce6105f862b&at=APcmpoy_i9t4xopBTuHlrgFo6s-T%3A1744656061111','approximator.weights')
approximator.weights = pickle.load(open('approximator.weights','rb'))
#final_scores = td_learning(env, approximator, num_episodes=10000, alpha=0.1, gamma=1, epsilon=0)

def get_action(state, score):
    env = Game2048Env()
    env.score = score
    env.board = state.copy()
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    next_values = []
    previous_score = env.score
    td_mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=5, gamma=1)
    root = TD_MCTS_Node(env.score,env=env,parent=None,chance=False)
    
    expansion = defaultdict(int)
    for _ in range(td_mcts.iterations):
        expansion[td_mcts.run_simulation(root,state)] += 1
    #print(expansion)
    
    # Select the best action (based on highest visit count)
    best_act = td_mcts.best_action_distribution(root)
    return best_act

if __name__ == "__main__":
    avg = 0
    for i in range(10):
        env = Game2048Env()
        state = env.reset()
        done = False
        score = 0
        while not done:
            state,score,done,_ = env.step(get_action(state,score))
            print(score, end="\r")
        print("Final score:",score)
        avg += score
    print("Average: ",avg/10)



