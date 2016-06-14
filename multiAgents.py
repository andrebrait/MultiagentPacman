# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
from game import Actions
import sys
import random, util
from math import isinf

from game import Agent
from pacman import GameState
from game import AgentState

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  initialFoodMap = None
  initialPosition = None
  initialCapsules = None

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.

    Parameters
    ----------
    currentGameState : pacman.GameState

    Returns
    -------
    float
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    curPos = currentGameState.getPacmanPosition()
    newFood = successorGameState.getFood().asList()
    oldCapsules = currentGameState.getCapsules()
    newGhostStates = successorGameState.getGhostStates()
    """:type : list[AgentState]"""

    if not self.initialFoodMap:
      self.initialFoodMap = util.Counter()
      self.initialPosition = currentGameState.getPacmanPosition()
      self.initialCapsules = currentGameState.getCapsules()
      oldFood = currentGameState.getFood().asList()
      for x in oldFood:
        self.initialFoodMap[x] = 1

    if successorGameState.isWin():
      return float('inf')

    if newPos == curPos:
      return float('-inf')

    minGhostDistance = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])

    if minGhostDistance <= 1:
      return float('-inf')

    if successorGameState.getPacmanPosition() in oldCapsules:
      return float('inf')

    if self.initialFoodMap[successorGameState.getPacmanPosition()] == 0 and successorGameState.getPacmanPosition() != self.initialPosition and successorGameState.getPacmanPosition() not in self.initialCapsules:
      return float('-inf')

    minNewFood = min([manhattanDistance(newPos, food) for food in newFood])

    retVal = (successorGameState.getScore() - currentGameState.getScore())*4 - minNewFood*4 + (0 if isinf(minGhostDistance) else minGhostDistance)

    "*** YOUR CODE HERE ***"
    return retVal

def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """

    lastIndex = gameState.getNumAgents() - 1

    def minimax(curState, depth, agentIndex):
      if depth == 0 or curState.isWin() or curState.isLose():
        return self.evaluationFunction(curState), None

      #Se o agente for o PacMan, procurar max
      if agentIndex == 0:
        retVal = float('-inf')
        retAction = None
        for action in [x for x in curState.getLegalActions(agentIndex) if x != Directions.STOP]:
          resultGhostActions = minimax(curState.generateSuccessor(agentIndex, action), depth - 1, agentIndex + 1)
          if resultGhostActions[0] > retVal:
            retVal = resultGhostActions[0]
            retAction = action
        return retVal, retAction
      #Se for um fantasma, fazer os min, ate que chegue o ultimo fantasma, no qual se passa para o pacman, que faz max
      else:
        retVal = float('inf')
        for action in [x for x in curState.getLegalActions(agentIndex) if x != Directions.STOP]:
          retVal = min(retVal, minimax(curState.generateSuccessor(agentIndex, action), depth if agentIndex < lastIndex else (depth - 1), (agentIndex + 1) if agentIndex < lastIndex else 0)[0])
        return retVal, None

    retVal = minimax(gameState, self.depth, 0)
    return retVal[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    lastIndex = gameState.getNumAgents() - 1

    def minimax(curState, depth, agentIndex, alpha, beta):
      if depth == 0 or curState.isWin() or curState.isLose():
        return self.evaluationFunction(curState), Directions.STOP

      # Se o agente for o PacMan, procurar max
      if agentIndex == 0:
        if alpha >= beta:
          return float('-inf'), Directions.STOP
        retVal = float('-inf')
        retAction = Directions.STOP
        for action in [x for x in curState.getLegalActions(agentIndex) if x != Directions.STOP]:
          resultGhostActions = minimax(curState.generateSuccessor(agentIndex, action), depth - 1, agentIndex + 1, alpha, beta)
          if resultGhostActions[0] > retVal:
            retVal = resultGhostActions[0]
            retAction = action
          alpha = max(alpha, retVal)
        return retVal, retAction
      # Se for um fantasma, fazer os min, ate que chegue o ultimo fantasma, no qual se passa para o pacman, que faz max
      else:
        if alpha >= beta:
          return float('inf'), Directions.STOP
        retVal = float('inf')
        retAction = Directions.STOP
        for action in [x for x in curState.getLegalActions(agentIndex) if x != Directions.STOP]:
          curResult = resultGhostActionsAndPacman = minimax(curState.generateSuccessor(agentIndex, action), depth if agentIndex < lastIndex else (depth - 1), (agentIndex + 1) if agentIndex < lastIndex else 0, alpha, beta)
          if curResult[0] < retVal:
            retVal = curResult[0]
            retAction = action
          beta = min(beta, retVal)
        return retVal, retAction

    retVal = minimax(gameState, self.depth, 0, float('-inf'),float('inf'))
    return retVal[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    lastIndex = gameState.getNumAgents() - 1

    def minimax(curState, depth, agentIndex):
      if depth == 0 or curState.isWin() or curState.isLose():
        return self.evaluationFunction(curState), Directions.STOP

      #Se o agente for o PacMan, procurar max
      if agentIndex == 0:
        retVal = float('-inf')
        retAction = Directions.STOP
        for action in [x for x in curState.getLegalActions(agentIndex) if x != Directions.STOP]:
          resultGhostActions = minimax(curState.generateSuccessor(agentIndex, action), depth - 1, agentIndex + 1)
          if resultGhostActions[0] > retVal:
            retVal = resultGhostActions[0]
            retAction = action
        return retVal, retAction
      #Se for um fantasma, fazer os min, ate que chegue o ultimo fantasma, no qual se passa para o pacman, que faz max
      else:
        retVal = float('inf')
        retAction = Directions.STOP
        expectedValue = 0.0
        legalGhostStates = [x for x in curState.getLegalActions(agentIndex) if x != Directions.STOP]
        numStates = len(legalGhostStates)
        for action in legalGhostStates:
          resultGhostActionsAndPacman = minimax(curState.generateSuccessor(agentIndex, action), depth if agentIndex < lastIndex else (depth - 1), (agentIndex + 1) if agentIndex < lastIndex else 0)
          if resultGhostActionsAndPacman[0] < retVal:
            retVal = resultGhostActionsAndPacman[0]
            retAction = action
          expectedValue += retVal/numStates
        return expectedValue, retAction

    retVal = minimax(gameState, self.depth, 0)
    return retVal[1]

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  curPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood().asList()
  newGhostStates = currentGameState.getGhostStates()
  """:type : list[AgentState]"""

  if currentGameState.isWin():
    return float('inf')

  minGhostDistance = min([manhattanDistance(curPos, ghostState.getPosition()) if ghostState.scaredTimer <= 0 else float('inf') for ghostState in newGhostStates])
  if minGhostDistance <= 1:
    return float('-inf')

  minNewFood = min([manhattanDistance(curPos, food) for food in newFood])

  retVal = currentGameState.getScore() - minNewFood + (100 if isinf(minGhostDistance) else minGhostDistance)

  "*** YOUR CODE HERE ***"
  return retVal

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

