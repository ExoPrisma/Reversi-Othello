# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("other_student_agent")
class OtherStudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(OtherStudentAgent, self).__init__()
    self.name = "OtherStudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    best_value = float('-inf')
    best_move = None

    end, _ , _ = check_endgame(chess_board, player, opponent)
    moves = get_valid_moves(chess_board, player)
    if end or not moves:
      return None

    for move in moves:
        simulated_board = deepcopy(chess_board)
        execute_move(simulated_board, move, player)

        # Evaluate opponent's responses (simulating the minimizing player)
        worst_opponent_value = float('inf')
        moves_opp = get_valid_moves(simulated_board, opponent)
        _, p_score , o_score = check_endgame(simulated_board, player, opponent)

        #Get top 10 best moves the opponent could make
        opponent_moves = sorted(moves_opp, key=lambda m: self.evaluate_board(simulated_board, opponent, p_score, o_score))[:10]

        for opponent_move in opponent_moves:
            opponent_board = deepcopy(simulated_board)
            execute_move(opponent_board, opponent_move, opponent)
            _, player_score, opponent_score = check_endgame(simulated_board, player, opponent)
            value_player = self.evaluate_board(opponent_board, player, player_score, opponent_score)

            # print("Opponent board score", value_opp)
            # print("Player board score", value_player)
          
            worst_opponent_value = min(worst_opponent_value, value_player)

        # Update best value for the maximizing player
        if worst_opponent_value > best_value:
            best_value = worst_opponent_value
            best_move = move

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    #return best
    return best_move 
  
  def evaluate_board(self, board, player, player_score, opponent_score):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == player) * 25
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - player) * -25

        # Mobility: the number of moves the opponent can make
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, 3 - player))
        mobility_score = player_moves - opponent_moves

        #X squares and C positions
        x_squares = [(1, 1), (1, board.shape[1] - 2), (board.shape[0] - 2, 1), (board.shape[0] - 2, board.shape[1] - 2)]
        c_positions = [(0, 1), (0, board.shape[1] - 2), (1, 0), (1, board.shape[1] - 1),
        (board.shape[1] - 2, 0), (board.shape[1] - 2, board.shape[1] - 1), 
        (board.shape[1] - 1, 1), (board.shape[1] - 1, board.shape[1] - 2)]

        num_moves_played = np.count_nonzero(board)  # Count the number of non-zero (occupied) cells
        early_game_threshold = 20  # Change this based on game observation
        x_score = 0
        c_score = 0
        x_penalty = 0
        c_penalty = 0

        if num_moves_played < early_game_threshold:
          # X-squares: Positions adjacent to corners (high negative weight early)
          # C-square heuristic (avoid early)
          x_score = -15 * sum(1 for pos in x_squares if board[pos] == player)
          c_score = -10 * sum(1 for pos in c_positions if board[pos] == player)
          x_penalty = 15 * sum(1 for pos in x_squares if board[pos] == 3 - player)
          c_penalty = 10 * sum(1 for pos in c_positions if board[pos] == 3 - player)
        else:
          x_score = -5 * sum(1 for pos in x_squares if board[pos] == player)
          c_score = -3 * sum(1 for pos in c_positions if board[pos] == player)
          x_penalty = 5 * sum(1 for pos in x_squares if board[pos] == 3 - player)
          c_penalty = 3 * sum(1 for pos in c_positions if board[pos] == 3 - player)

        # Stability (stable discs) and edge stability
        stable_score = 0
        # Simple idea: assume discs at edges and corners are more stable
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if r == 0 or r == board.shape[0] - 1 or c == 0 or c == board.shape[1] - 1:
                    if board[r, c] == player:
                        stable_score += 5
                    elif board[r, c] == 3 - player:
                        stable_score -= 5

        # Combine scores
        total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score + x_score + x_penalty + c_score + c_penalty + stable_score
        return total_score
  
  




  
  

