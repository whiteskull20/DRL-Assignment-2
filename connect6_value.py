import sys
import numpy as np
import random
import copy
import math
from collections import defaultdict
from tqdm import trange
import pickle
from table import feature_table, line_table

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        selected = random.sample(empty_positions, 1)
        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return
    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")
class Connect6GameEnv:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        return self.board
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        return self.board
    
    def check_win(self,position=None):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        for p in position:
            for l in line_table[p]:
                if sum([1 for x in l if self.board[x] == 1]) == 6:
                    return 1
                if sum([1 for x in l if self.board[x] == 2]) == 6: 
                    return 2
        return 0
    def check_instant_win(self,move_left,position=None):
        """Checks if a player has instant win
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        if len(position) == 0:
            position = [(i//19,i%19) for i in range(19*19) if self.board[i//19,i%19] == self.turn]
        threat = []
        for p in position:
            for l in line_table[p]:
                f = [self.board[x] for x in l]
                cnt1 = sum([1 for x in f if x == self.turn])
                cnt2 = sum([1 for x in f if x == 3-self.turn])
                if cnt1 >= 6-move_left and cnt2 == 0:
                    threat += [l[i] for i in range(6) if (f[max(i-1,0)] != 0 or f[min(i+1,5)] != 0) and f[i] == 0 ]
        return list(set(threat))
    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')
    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)
    def step(self, move):
        """Places stones and checks the game status."""
        color = 'b' if self.turn == 1 else 'w'
        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        return  self.board,self.check_win(position=positions)
    def get_legal_moves(self):
        return [f"{self.index_to_label(c)}{r+1}" for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
    def generate_move(self, color):
        """Generates a random move for the computer."""
        return
class NTupleApproximator:
    def __init__(self, board_size, pattern_len):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.pattern_len = pattern_len
        self.weights = defaultdict(float)

    def get_feature(self, board, coord, direction): # unused
        rmap = [0,2,1]
        c = coord
        if coord[0] > self.board_size - direction[0]*self.pattern_len or coord[1] > self.board_size - direction[1]*self.pattern_len or coord[1] + direction[1]*self.pattern_len < -1 or coord[0] + direction[0]*self.pattern_len < -1:
            return None
        feature = [ board[coord[0]+_*direction[0],coord[1]+_*direction[1]] for _ in range(self.pattern_len)]
        return tuple([ int(x) for x in feature])
    def get_feature_weight(self,feature):
        rmap = [0,2,1]  # Precompute reverse mapping for features
        reversed_feature = tuple(rmap[x] for x in feature)
        # Combine evaluation directly without redundant loops
        weights = [self.weights[feature], self.weights[feature[::-1]],
                -self.weights[reversed_feature], -self.weights[reversed_feature[::-1]]]
        return sum(weights)
    def value(self, board, player,last_board=None,last_value=0,last_turn=0):
        value = 0
        directions = [(1, 0), (0, 1), (1, 1), (1,-1)]  # Predefined directions
        #print(last_board,last_value,last_turn,file=sys.stderr)
        if last_board is not None: # calculate value based on previous boards to save time
            mask = [ (i//19,i%19) for i in range(19*19) if board[i//19,i%19] != last_board[i//19,i%19] ]
            
            #print(board,last_board,file=sys.stderr)
            #print(mask,file=sys.stderr)
            if last_turn != player:
                value = -last_value
            else:
                value = last_value
            for feature in feature_table:    
                if set(feature) & set(mask):
                    f1,f2 = tuple([board[_] for _ in feature]), tuple([last_board[_] for _ in feature])
                    if f1 is None:
                        continue
                    w = 1 if player == 1 else -1
                    value += w * (self.get_feature_weight(f1) - self.get_feature_weight(f2))*sum(1 for x in feature if x != 0)
            #assert(False)
            return value
        for f in feature_table:
            feature = tuple([board[_] for _ in f])
            w = 1 if player == 1 else -1
            w2 = sum(1 for x in feature if x != 0)
            value += w * self.get_feature_weight(feature)*w2
        
        return value


    def update(self, board, delta, alpha,player):
        # Update weights based on the error.
        value = 0
        
        for feature in feature_table:
            f = tuple([board[_] for _ in feature])
            rf =  tuple([ 3 - _ if _ != 0 else 0 for _ in f])
            w2 = sum(1 for x in f if x != 0) # higher update weight priority for larger number of stones (which is more important)
            for F,w in zip([f,f[::-1],rf,rf[::-1]],[1,1,-1,-1] if player == 1 else [-1,-1,1,1]):
                self.weights[F] += delta*alpha*w/len(feature_table)*w2
        return value
def td_learning(env: Connect6GameEnv, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=1):
    """
    Actually its MC
    """
    final_scores = []
    delta_avg = []
    for episode in trange(num_episodes):
        state = env.reset()
        state = state.copy()
        trajectory = []  # Store trajectory data if needed
        done = False
        moves_left = 1
        player = 1
        while not done:
            env.turn = player
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                break     
            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                next_values = []
                for a in legal_moves:
                    sim_env = copy.deepcopy(env)
                    sim_env.turn = player
                    sim_state , win = sim_env.step(a)
                    if win == player:
                        next_values.append(1)
                    elif win != 0:
                        next_values.append(-1)
                    else:
                        next_values.append(approximator.value(sim_state,player))
                maxval = np.max(next_values)
                action = (legal_moves[random.choice([a for a, v in enumerate(next_values) if v == maxval])])
            trajectory.append((state,player))
            moves_left -= 1
            if moves_left == 0:
                player = 3 - player
                moves_left = 2
            next_state, done = env.step(action)
            state = next_state
            state = state.copy()
        trajectory.append((state,player)) # terminal state included as well
        delta_sum = 0
        reward = 1
        for state, player in reversed(trajectory):
            if done == 0:
                w = 0
            else:
                w = (1 if done == player else -1)
            delta = reward * w - approximator.value(state,player)
            delta_sum += delta
            approximator.update(state, delta, alpha,player)
            reward *= gamma
        delta_avg.append(delta_sum/len(trajectory))
        if (episode + 1) % 10 == 0:
            avg_delta = np.mean(delta_avg[-100:])
            print(f"Episode {episode+1}/{num_episodes}  | Avg delta: {avg_delta} | Epsilon = {epsilon}")
        epsilon *= 0.999
        epsilon = max(0.01,epsilon)
        if (episode + 1) % 100 == 0:
          # Backup the approximator's weights
          with open('approximator_6_epsgreed.weights', 'wb') as f: # backup in case of failure
              pickle.dump({k : v for k,v in approximator.weights.items()}, f)
    return final_scores
if __name__ == "__main__":
    env = Connect6GameEnv()
    approximator = NTupleApproximator(19, 8)
    approximator.weights = pickle.load(open('approximator_6_epsgreed.weights.backup', 'rb'))
    approximator.weights = defaultdict(lambda: 0.0,{tuple(int(_) for _ in k): v for k,v in approximator.weights.items()})
    td_learning(env, approximator, num_episodes=10000, alpha=0.01, gamma=0.9, epsilon=0.5)

