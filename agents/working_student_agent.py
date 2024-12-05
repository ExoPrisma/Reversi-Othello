# Student agent: Add your own agent here
import numbers
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("working_student_agent")
class WorkingStudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(WorkingStudentAgent, self).__init__()
    self.name = "StudentAgent"

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
    # print("Step")

    start_time = time.time()
    time_limit = 1.7    # Buffer under 2s

    empty_squares = np.sum(chess_board == 0)
    board_size = chess_board.shape[0]
    total_squares = board_size**2
    
    if empty_squares > int(total_squares * 0.9):      # Early game
      depth = board_size // 2        
      scale = 1
    elif empty_squares > int(total_squares * 0.75):   # Mid game
      depth = (2 * board_size) // 3  
      scale = 2
    else:                                             # Late game
      depth = board_size             
      scale = 5

    # Initialize root node
    root = Node(chess_board, player, None)

    # Iterative simulations until time runs out
    while time.time() - start_time < time_limit:
      self._perform_simulations(root, player, opponent, depth, start_time, time_limit, scale)

    # print(time.time() - start_time)

    # print(root)
    best_child = self._select_best_move(root, player, scale)
    if best_child:
      # print(time.time() - start_time)
      return best_child.move
    else:
      print("No valid moves available. Passing turn.")
      return None
    
  def _perform_simulations(self, root, player, opponent, depth, start_time, time_limit, scale):
    """Wrapper function for MCT steps"""

    if time.time() - start_time >= time_limit:
      return

    leaf = self._select(root) 
    
    if not leaf.is_terminal():  # If it's not a leaf node
      child = self._expand(leaf, player, scale)
      if time.time() - start_time >= time_limit:
        return
      reward = self._simulate(child, player, opponent, depth, start_time, time_limit)
      self._backpropagate(child, reward)
    else:  # If it's a leaf node
      reward = self._simulate(leaf, player, opponent, depth, start_time, time_limit)
      self._backpropagate(leaf, reward)

  def _select(self, node):
    """Select the most promising child using UCT and alpha-beta pruning."""
    # print("Selecting...")

    while node.children:
      node = max(node.children, key=lambda x: x.uct())
    return node
    
  def _expand(self, node, player, scale):
    """Expand all valid moves from the current node."""
    # print("Expanding...")
    valid_moves = get_valid_moves(node.state, player)

    if not valid_moves:
      return node
    
    moves_with_scores = []
    for move in valid_moves:
      corner_score = self._corner_score(move, node.state, 1)
      mobility_score = self._mobility_score(node.state, move, player, 1)
      greed_penalty = self._greed_penalty(node.state, player, move, scale)
      
      combined_score = (corner_score + mobility_score - greed_penalty)

      moves_with_scores.append((move, combined_score))
    
    moves_with_scores.sort(key=lambda x: x[1], reverse=True)

    for move, _ in moves_with_scores[:3]:
      new_state = deepcopy(node.state)
      execute_move(new_state, move, player)
      new_node = Node(new_state, 3 - player, move, parent=node)
      node.children.append(new_node)

    # print(f"Expanded node with {len(node.children)} children.")
    return node.children[0] if node.children else node
  
  def _simulate(self, node, player, opponent, depth, start_time, time_limit):
    """
    Simulate a game from the given node using a semi-random strategy.
    Moves that lead to very bad positions are discouraged during rollouts
    """
    # print("Simulating...")

    current_state = deepcopy(node.state)
    current_player = player
    current_depth = 0
    board_size = node.state.shape[0]

    while current_depth < depth:
      if time.time() - start_time >= time_limit:
        break
      
      is_endgame, score1, score2 = check_endgame(current_state, current_player, opponent)

      if is_endgame:
        # Return the reward based on the simulation result
        return 1 if score1 > score2 else 0 if score1 < score2 else 0

      valid_moves = get_valid_moves(current_state, current_player)
      corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
      corner_moves = [m for m in valid_moves if m in corners]

      if corner_moves:
        move = corner_moves[np.random.randint(len(corner_moves))]
        execute_move(current_state, move, current_player)
      else:
        if valid_moves:
          move = valid_moves[np.random.randint(len(valid_moves))]
          execute_move(current_state, move, current_player)
        else:
          current_player = 3 - current_player  # Pass turn if no valid moves

      current_player = 3 - current_player
      current_depth += 1

    node.visits += 1
    # If the simulation doesn't reach the endgame, return a neutral score
    return 0  # Neutral score for incomplete simulations
  
  def _backpropagate(self, node, reward):
    """Backpropagate the reward up the tree."""
    # print("Backpropagating...")

    while node:
      node.wins += reward

      node.visits += 1
      node = node.parent

  # Heuristic move choice (See https://royhung.com/reversi)
  def _select_best_move(self, node, player, scale):
    """Select the best move using adjusted UCT scores and domain knowledge given time in game"""
    best_score = float('-inf')
    best_child = None

    for child in node.children:
      # Adjusted UCT score calculation
      uct = child.uct()
      inner_score = self._inner_score(uct, child.state, child.move, scale)
      corner_score = self._corner_score(child.move, child.state, 1)
      greed_penalty = self._greed_penalty(child.state, player, child.move, scale)
      position_penalty = self._position_penalty(child.state, uct, child.move, scale)
      mobility_score = self._mobility_score(child.state, child.move, player, 1)

      # variables = {
      #   "uct": uct,
      #   "inner_score": inner_score,
      #   "corner_score": corner_score,
      #   "mobility_score": mobility_score,
      #   "greed_penalty": greed_penalty,
      #   "position_penalty": position_penalty
      # }
      # print("Score")

      # for name, value in variables.items():
      #   if isinstance(value, numbers.Number):
      #       print(f"{name} is a number: {value}")
      #   else:
      #       print(f"{name} is NOT a number: {value}")

      adjusted_score = (
        uct + 
        inner_score + 
        corner_score +
        mobility_score -
        greed_penalty - 
        position_penalty
      )

      if adjusted_score > best_score:
        best_score = adjusted_score
        best_child = child

    return best_child

  # Heuristics 
  # Idea from website (See https://royhung.com/reversi)
  def _inner_score(self, uct, board, move, scale):
    """Encourage moves closer to the center."""
    i, j = move
    middle = ((board.shape[0] - 1) / 2)
    a = 5 / scale # Tunable parameter - Decrease

    numerator = a * uct
    denomator = np.sqrt((i - middle)**2 + (j - middle)**2)

    return numerator / denomator
  
  # Idea from YouTube video (See https://youtu.be/sJgLi32jMo0?si=RLyWJO_KTgRF7A2G)
  def _corner_score(self, move, board, scale):
    """Evaluate the move based on its proximity to corners."""
    b = 100 * scale # Tunable parameters - Increase
    
    corners = [
      (0, 0), 
      (0, board.shape[1] - 1), 
      (board.shape[0] - 1, 0), 
      (board.shape[0] - 1, board.shape[1] - 1)
    ]

    return b if move in corners else 0
  
  # Idea from chat gpt
  def _mobility_score(self, board, move, player, scale):
    """
    Calculate the mobility score for a given move.
    Encourages moves that limit the opponent's options while maintaining flexibility.
    """
    e = 5 / scale # Tunable constant - Decrease

    # Simulate the move
    simulated_board = deepcopy(board)
    execute_move(simulated_board, move, player)

    opponent = 3 - player
    opponent_moves = len(get_valid_moves(simulated_board, opponent))

    my_moves = len(get_valid_moves(simulated_board, player))

    return e * (my_moves - opponent_moves)

  # Idea from website (See https://royhung.com/reversi)
  def _greed_penalty(self, board, player, move, scale):
    """Discourage flipping too many discs early in the game."""
    c = 1 / scale # Tunable parameter - Decrease

    total_discs = np.sum(board > 0)

    player_captures = count_capture(board, move, player)
    
    opponent = 3 - player
    opponent_moves = get_valid_moves(board, opponent)
    opponent_captures = max(
      (count_capture(board, m, opponent) for m in opponent_moves),
      default=0
    )

    t = 1 if player == 1 else -1
    greed_penalty = t * (player_captures - opponent_captures) / (total_discs + 1e-6)**c  # Avoid division by zero

    return greed_penalty
  
  # Idea from website (See https://royhung.com/reversi)
  def _position_penalty(self, board, uct, move, scale):
    """Discourage moves near corners or edges."""
    d, e = 1 / scale , 0.01 / scale # Tunable parameter - Decrease

    corners = [
        (0, 0),
        (0, board.shape[1] - 1),
        (board.shape[0] - 1, 0),
        (board.shape[0] - 1, board.shape[1] - 1)
    ]
    x_squares = [
        (corner[0] + 1, corner[1] + 1) for corner in corners
    ]
    c_squares = [
        (corner[0], corner[1] + 1) if corner[1] + 1 < board.shape[1] else None
        for corner in corners
    ] + [
        (corner[0] + 1, corner[1]) if corner[0] + 1 < board.shape[0] else None
        for corner in corners
    ]

    c_squares = [sq for sq in c_squares if sq is not None]

    if move in x_squares:
        return d * uct
    if move in c_squares:
        return e * uct
    return 0  

class Node:
  def __init__(self, state, player, move, parent=None):
    self.state = state          # Game state at node
    self.player = player        # Player whose turn it is
    self.move = move            # Move that led to this state
    self.parent = parent        # Reference to the parent node
    self.children = []          # List of child nodes
    self.visits = 0             # Total visits to this node
    self.wins = 0               # Number of wins from simulations through this node
    self.mobility = 0           # Mobility score: sum of valid moves

  def is_terminal(self):
    """Leaf (Terminal) check"""
    valid_moves = get_valid_moves(self.state, self.player)
    return len(valid_moves) == 0

  def uct(self):
    """Upper Confidence Bound for Trees."""
    c = np.sqrt(2)
    if self.visits == 0:
      return float('inf')
    
    return self.wins / self.visits + (c * np.sqrt(2 * np.log(self.parent.visits) / self.visits))

  def __str__(self):
    """String representation of the node, printing useful information."""
    # Print the node's state, player, move, number of visits, and value
    state_str = str(self.state)  # You can format this as you like to show the board
    return (f"Node(State:\n{state_str}\nPlayer: {self.player}, "
            f"Move: {self.move}, Wins: {self.wins}, Mobility: {self.mobility}, "
            f"Visits: {self.visits})")