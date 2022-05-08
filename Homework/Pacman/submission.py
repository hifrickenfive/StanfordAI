from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState

# Add extra imports
import math

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIdx):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIdx, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current GameState (defined in pacman.py)
    and a proposed action and returns a rough estimate of the resulting successor
    GameState's value.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Don't forget to limit the search depth using self.depth. Also, avoid modifying
      self.depth directly (e.g., when implementing depth-limited search) since it
      is a member variable that should stay fixed throughout runtime.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIdx):
        Returns a list of legal actions for an agent
        agentIdx=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIdx, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def GetMinMaxAction(gameState, agentIdx, depth):

      # Base case
      if gameState.isLose() or gameState.isWin(): 
        return (gameState.getScore(), Directions.STOP)
      elif depth == 0:
        return (self.evaluationFunction(gameState), Directions.STOP)
      # Recurse
      else:
        # Evaluate nextAgentIdx
        nextAgentIdx = agentIdx + 1
        if nextAgentIdx == gameState.getNumAgents():
          nextAgentIdx = self.index
          depth -= 1

        # Evaluate next actions and scores
        actions = gameState.getLegalActions(agentIdx)
        nextScores  = []
        for action in actions:
          nextGameState = gameState.generateSuccessor(agentIdx, action)
          nextScore, _  = GetMinMaxAction(nextGameState, nextAgentIdx, depth)
          nextScores.append(nextScore)

        # Max action by pacman
        if agentIdx == self.index:
            maxVal = max(nextScores)
            maxIdx = nextScores.index(maxVal)
            maxAction = actions[maxIdx]
            return (maxVal, maxAction)
        # Min action by opponents
        else:
            minVal = min(nextScores)
            minIdx = nextScores.index(minVal)
            minAction = actions[minIdx]
            return (minVal, minAction)

    return GetMinMaxAction(gameState, self.index, self.depth)[1]

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    initAlpha = -math.inf
    initBeta = math.inf

    def GetMinMaxAction(gameState, agentIdx, depth, alpha, beta):

      # Base case
      if gameState.isLose() or gameState.isWin(): 
        return (gameState.getScore(), Directions.STOP)
      elif depth == 0:
        return (self.evaluationFunction(gameState), Directions.STOP)
      # Recurse
      else:
        # Evaluate nextAgentIdx
        nextAgentIdx = agentIdx + 1
        if nextAgentIdx == gameState.getNumAgents():
          nextAgentIdx = self.index
          depth -= 1

        # Max action by pacman
        if agentIdx == self.index:
          # Initialise
          maxVal = -math.inf
          maxAction = Directions.STOP
          # Evaluate next actions and scores
          actions = gameState.getLegalActions(agentIdx)
          for action in actions:
            nextGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore, _  = GetMinMaxAction(nextGameState, nextAgentIdx, depth, alpha, beta)
            if nextScore > maxVal:
              maxVal = nextScore
              maxAction = action
            # Update alpha
            alpha = max(nextScore, alpha)
            # Check early terminate condition
            if beta <= alpha:
              break
          return (maxVal, maxAction)

        # Min action by opponents
        else:
          # Initialise
          minVal = math.inf
          minAction = Directions.STOP
          # Evaluate next actions and scores
          actions = gameState.getLegalActions(agentIdx)
          for action in actions:
            nextGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore, _  = GetMinMaxAction(nextGameState, nextAgentIdx, depth, alpha, beta)
            if nextScore < minVal:
              minVal = nextScore
              minAction = action
            # Update beta
            beta = min(nextScore, beta)
            # Check early terminate condition
            if beta <= alpha:
              break
          return (minVal, minAction)
    return GetMinMaxAction(gameState, self.index, self.depth, initAlpha, initBeta)[1]

    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def GetExpectiMaxAction(gameState, agentIdx, depth):

      # Base case
      if gameState.isLose() or gameState.isWin(): 
        return (gameState.getScore(), Directions.STOP)
      elif depth == 0:
        return (self.evaluationFunction(gameState), Directions.STOP)
      # Recurse
      else:
        # Evaluate nextAgentIdx
        nextAgentIdx = agentIdx + 1
        if nextAgentIdx == gameState.getNumAgents():
          nextAgentIdx = self.index
          depth -= 1

        # Evaluate next actions and scores
        actions = gameState.getLegalActions(agentIdx)
        nextScores  = []
        for action in actions:
          nextGameState = gameState.generateSuccessor(agentIdx, action)
          nextScore, _  = GetExpectiMaxAction(nextGameState, nextAgentIdx, depth)
          nextScores.append(nextScore)

        # Max action by pacman
        if agentIdx == self.index:
            maxVal = max(nextScores)
            maxIdx = nextScores.index(maxVal)
            maxAction = actions[maxIdx]
            return (maxVal, maxAction)
        # Random action by opponents
        else:
            averageVal = sum(nextScores)/len(actions)
            randAction = random.choice(actions)
            return (averageVal, randAction)

    return GetExpectiMaxAction(gameState, self.index, self.depth)[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """

  # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
  
  # Get state information
  score = currentGameState.getScore()
  pacmanPosition = currentGameState.getPacmanPosition() # tuple (x, y)
  food = currentGameState.getFood() # object
  numFood = currentGameState.getNumFood() 
  ghostStates = currentGameState.getGhostStates() # str(ghostStates[1]) = 'Ghost: (x,y)=(10.0, 4.0), South'
  ghostPositions = currentGameState.getGhostPositions() # list of tuples (x, y) positions
  capsules = currentGameState.getCapsules() # list of tuples (x, y) positions

  # Set params
  evadeThreshold = 2

  # Initialise flags
  ghostScared = False
  ghostTooClose = False
  numPillsEqualFood = False

  ################################# Evaluate raw inputs  ############################
  # Food distance
  foodList = list(food)
  foodScore = []
  foodDistances = []
  for i in range(food.width): # x axis
      for j in range(food.height): # y axis
          if foodList[i][j]:
              foodDistance = util.manhattanDistance(pacmanPosition,(i,j))
              foodDistances.append(foodDistance)
              foodScore.append(1/foodDistance**2)

  # Ghost Distance, Ghost timer
  ghostDistances = []
  for ghost in ghostStates:
    ghostDistance = util.manhattanDistance(pacmanPosition, ghost.getPosition())
    ghostDistances.append(ghostDistance)
    if ghost.scaredTimer > 0:
      ghostScared = True
    if min(ghostDistances) < evadeThreshold:
      ghostTooClose = True
      
  # Capsule distance
  capsuleScore = []
  for capsule in capsules:
    capsuleDistance = util.manhattanDistance(pacmanPosition, capsule)
    capsuleScore.append(1/capsuleDistance**2)

  ################################# Create Scores ############################
  ghostScore = 0
  for idx, ghostDistance in enumerate(sorted(ghostDistances)):
    # ghostScore += 1/(idx+1) * math.exp(ghostDistance-evadeThreshold) / (1 + math.exp(ghostDistance-evadeThreshold))
    # ghostScore += (1/(idx+1))*(1/ghostDistance**2)
    ghostScore += (1/ghostDistance)


  capsuleScore = 0
  for capsule in capsules:
    capsuleDistance = util.manhattanDistance(pacmanPosition, capsule)
    capsuleScore += 1/capsuleDistance**2
  
  ################################# State transitions ############################
  if ghostScared:
    pacState = 'KILL'
  elif ghostTooClose:
    pacState = 'EVADE'
  else:
    pacState = 'EAT'

  ################################# Evaluation Function ##########################
  if pacState == 'EVADE':
    return score - 160*ghostScore
  elif pacState == 'KILL':
    return score + 200*ghostScore
  elif pacState == 'EAT':
    return score + 10*sum(foodScore) - 12*ghostScore + 10*capsuleScore

  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
