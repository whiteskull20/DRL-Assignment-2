import sys
import numpy as np
import random
import copy
import math
from collections import defaultdict
from connect6_value import Connect6GameEnv, NTupleApproximator
import pickle
from tqdm import trange
from connect6_mcts import MCTS, MCTS_Node
import cProfile
class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.approximator = NTupleApproximator(19, 8)
        
        self.approximator.weights = pickle.load(open('approximator_6_epsgreed.weights.backup','rb'))
        self.approximator.weights = defaultdict(lambda: 0.0,{tuple(int(_) for _ in k): v for k,v in self.approximator.weights.items()})

        print(sorted(self.approximator.weights.items(),key=lambda x:x[1],reverse=True)[:20],file=sys.stderr)
        print(sorted(self.approximator.weights.items(),key=lambda x:x[1],reverse=True)[-20:],file=sys.stderr)
        self.mcts = MCTS( self.approximator, iterations=5000, simulation_batch=1, rollout_depth=0, exploration_constant=1.41, gamma=0.999)
        self.remain_move = 1

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        self.remain_move = 1
        print("= ", flush=True)
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.remain_move = 1
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
        self.remain_move -= len(stones)
        if self.remain_move == 0:
            self.turn = 3 - self.turn
            self.remain_move = 2
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        profiler = cProfile.Profile()
        profiler.enable()
        estimate = self.approximator.value(self.board,self.turn)
        print("Current estimate =",estimate,file=sys.stderr)
        if self.game_over:
            print("? Game over")
            return
        env = Connect6GameEnv(self.size)
        env.board = self.board
        env.turn = self.turn
        
        root = MCTS_Node(self.remain_move, env.turn)
        # Run multiple simulations to build the MCTS tree
        expansion = defaultdict(int)
        for _ in trange(self.mcts.iterations):
            expansion[self.mcts.run_simulation(root,env,self.board,self.turn,estimate)] += 1
        print("Expansion Count:",expansion,file=sys.stderr)
        maxval = -1e10
        action = None
        for c in root.children.values():
            if c.visits == 0:
                continue
            w = -1 if c.turn != root.turn else 1
            if maxval < w*c.total_reward / c.visits:
                maxval = w*c.total_reward / c.visits
                action = c.action
        #print(self.board,file=sys.stderr)
        #print('',file=sys.stderr)
        print(action,root.children[action].total_reward/root.children[action].visits,root.children[action].turn,root.move_left, root.turn, file=sys.stderr)
        '''
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            print("? Game over")
            return
        next_values = []
        for a in legal_moves:
            sim_env = copy.deepcopy(env)
            sim_state , win = sim_env.step(a)
            if win == self.turn:
                next_values.append(1000000000)
            elif win != 0:
                next_values.append(-1000000000)
            else:
                next_values.append(self.approximator.value(sim_state,self.turn))
            #print(a,next_values[-1],file=sys.stderr)
        maxval = np.max(next_values)
        #print(maxval,file=sys.stderr)
        action = (legal_moves[random.choice([a for a, v in enumerate(next_values) if v == maxval])])'''
        self.play_move(color, action)
        #print(action,file=sys.stderr)
        print(f"{action}\n\n", end='', flush=True)
        
        
        profiler.disable()
        profiler.dump_stats(f"profile.prof")
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
            line = sys.stdin.readline()
            if not line:
                break
            self.process_command(line)

if __name__ == "__main__":
    game = Connect6Game()
    
    game.run()


