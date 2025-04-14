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
def step_chance(env,action):
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
    config = []
    pconfig = []
    if moved:
        empty_cells = list(zip(*np.where(self.board == 0)))
        P = 1/len(empty_cells)
        for x,y in empty_cells:
            self.board[x,y] = 2
            config.append(self.board.copy())
            pconfig.append(0.9*P)
            self.board[x,y] = 4
            config.append(self.board.copy())
            pconfig.append(0.1*P)
            self.board[x,y] = 0
    return config, pconfig
class TD_MCTS_Node:
    def __init__(self,env, score, parent=None, action=None, chance=True,prob=1):
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
        self.visits = 0
        self.total_reward = 0.0
        self.chance = chance # whether it is chance node
        self.state = None
        self.prob = prob
        if self.chance:
            self.config, self.pconfig = step_chance(env,action)
            env2 = copy.deepcopy(env)
            for i in range(len(self.config)):
                env2.state = self.config[i]
                self.children[i] = TD_MCTS_Node(env2,score,parent=self,chance=False,prob=self.pconfig[i])
        else:
            # List of untried actions based on the current state's legal moves
            self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
            self.state = env.board.copy()

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=1):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        maxchild = None
        maxucb = -10000
        if not node.chance: # sample by UCB
            for c in node.children.values():
                ucb = c.total_reward / c.visits + self.c * math.sqrt(math.log(node.visits) / c.visits)
                if ucb > maxucb:
                    maxucb = ucb
                    maxchild = c
            return maxchild
        else: # sample by pconfig
            return node.children[random.choices(len(self.config),self.pconfig,k=1)]

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


    def run_simulation(self, root,state):
        node = root
        sim_env = self.create_env_from_state(state,node.score)
        while node.fully_expanded():
            node = self.select_child(node) #chance node
            node = self.select_child(node) #non chance node
        sim_env = self.create_env_from_state(node.state,node.score)   
        # TODO: Expansion: if the node has untried actions, expand one.
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            sim_env2 = copy.deepcopy(sim_env)
            state, reward, done, _ = sim_env2.step(action) #simulate step just to get the reward
            if done: # terminal, count as 1 simulation
                self.backpropagate(node, reward)
            else:
                node.children[action] = TD_MCTS_Node(sim_env,reward, parent=node, action=action,chance=True) # create chance node
                node.untried_actions.remove(action)
                node = self.select_child(node.children[action]) # sample one state
                sim_env = self.create_env_from_state(node.state,node.score)
        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


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

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    delta_avg = []
    for episode in trange(num_episodes):
        state = env.reset()
        state = state.copy()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                next_values = []
                for a in legal_moves:
                  sim_env = copy.deepcopy(env)
                  sim_env.board = state.copy()
                  sim_env.score = previous_score
                  sim_state , sim_score, _, _ = sim_env.step(a)
                  next_values.append(sim_score-previous_score+gamma*approximator.value(sim_state))
                maxval = np.max(next_values)
                action = legal_moves[random.choice([a for a, v in enumerate(next_values) if v == maxval])]
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))
            # TODO: Store trajectory or just update depending on the implementation
            trajectory.append((state,incremental_reward,next_state.copy()))
            # TODO: TD-Learning update
            state = next_state
            state = state.copy()
        #print(trajectory)
        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        delta_sum = 0
        for state, reward, next_state in reversed(trajectory):
            delta = reward + gamma * approximator.value(next_state) - approximator.value(state)
            delta_sum += delta
            approximator.update(state, delta, alpha)
        delta_avg.append(delta_sum/len(trajectory))
        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            avg_delta = np.mean(delta_avg[-100:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Avg delta: {avg_delta}")
        if (episode + 1) % 10000 == 0:
          # Backup the approximator's weights
          with open('approximator_6.weights', 'wb') as f: # backup in case of failure
              pickle.dump(approximator.weights, f)
    return final_scores


import requests
approximator = NTupleApproximator(board_size=4)
def download_public_file(file_url, destination_path):
    response = requests.get(file_url)
    response.raise_for_status()  # Ensure the request was successful
    
    with open(destination_path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully to {destination_path}")
download_public_file('https://drive.google.com/file/d/1YzHJwp5Kx8F0ueHXOALY44Zz4KCRclcr/view?usp=sharing','approximator.weights')
approximator.weights = pickle.load(open('approximator.weights','rb'))
#final_scores = td_learning(env, approximator, num_episodes=10000, alpha=0.1, gamma=1, epsilon=0)

def get_action(state, score):
    env = Game2048Env()
    env.score = score
    env.board = state.copy()
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    next_values = []
    previous_score = env.score
    for a in legal_moves:
      sim_env = copy.deepcopy(env)
      sim_env.board = state.copy()
      sim_env.score = previous_score
      sim_env.step(a)
      next_values.append(sim_env.score-previous_score+approximator.value(sim_env.board))
    maxval = np.max(next_values)
    action = legal_moves[random.choice([a for a, v in enumerate(next_values) if v == maxval])]
    return action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


