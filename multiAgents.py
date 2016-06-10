# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# touch /tmp/pacmeh; for i in $(seq 1 100);do python pacman.py --frameTime 0 -p ReflexAgent -k1 -g DirectionalGhost >> /tmp/pacmeh; done; echo "Loss: "$(cat /tmp/pacmeh | grep Loss | wc -l); echo "Win: "$(cat /tmp/pacmeh | grep "Record.*Win" | wc -l); rm /tmp/pacmeh;

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

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

    #print gameState.getPacmanPosition()
    #print scores

    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    #chosenIndex=bestIndices[0]

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
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    curPos = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newFood = successorGameState.getFood()
    foodList = newFood.asList();

    #print "eval func"
    #print newPos
    #print oldFood
    #print newGhostStates
    #print newScaredTimes

    "*** YOUR CODE HERE ***"

    """
    minDist=float('inf')
    foodDist=0
    ghostDist=float('inf')
    x=y=0
    for line in oldFood:
      for f in line:
        if f:
          dist=manhattanDistance(newPos,(x,y))
          foodDist+=dist
          minDist=min(dist,minDist)
          y+=1
      x+=1

    # se o fantasma esta vulneravel, vai filhao
    gc=0
    for s in newScaredTimes:
      if s > 0:
        return float('inf')
      else:
        gDist=manhattanDistance(newPos,newGhostStates[gc].getPosition())
        ghostDist=min(ghostDist,gDist)
      gc+=1
    # Corre Berg
    if ghostDist < 3:
      return ghostDist
    #Come bolinha mais longe dos fantasmas
    return (foodDist+gDist)
    """

    """
    #if newPos in [(8,5),(9,5),(10,5),(11,5)]:
    #  return float('-inf')
    ghostDist=float('inf')
    gc=0
    for s in newScaredTimes:
      if s > 0:
        return float('inf')
      else:
        gDist=manhattanDistance(newPos,newGhostStates[gc].getPosition())
        ghostDist=min(ghostDist,gDist)
      gc+=1

    if ghostDist < 5:
      return ghostDist

    minDist=float('inf')
    x=y=0
    for line in food:
      for f in line:
        if f:
          dist=manhattanDistance(newPos,(x,y))
          minDist=min(dist,minDist)
          y+=1
      x+=1

    if newPos == curPos:
      return float('-inf')
    else:
      return float('inf')+minDist
    """

    if successorGameState.isWin():
      return float('inf')

    if newPos == curPos:
      return float('-inf')

    minGhostDist=float('inf')
    ghostDist=0
    for ghost in newGhostStates:
      dist=manhattanDistance(newPos,ghost.getPosition())
      minGhostDist=min(minGhostDist,dist)
      ghostDist+=dist

    if minGhostDist < 2:
      return float('-inf')

    foodDist=0
    minFoodDist=float('inf')
    for food in foodList:
      dist=manhattanDistance(food,newPos)
      minFoodDist=min(minFoodDist,dist)
      foodDist+=dist

    value=currentGameState.getScore()
    value-=minFoodDist
    if currentGameState.getNumFood() > successorGameState.getNumFood():
      value+=foodDist

    return value
    #return (ghostDist + minFoodDist + foodDist)

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
    "*** YOUR CODE HERE ***"
    def minimax(state,depth,idAgent):
      if (depth == 0) or state.isWin() or state.isLose():
        return (self.evaluationFunction(state),'meh')
      #max
      if idAgent == 0:
        value=float('-inf')
        actionC=None
        for action in state.getLegalActions(0):
          result=minimax(state.generateSuccessor(0, action),depth-1,1)
          if result[0] > value:
            value=result[0]
            actionC=action
        return (value,actionC)
      #min
      else:
        value=float('inf')
        ghost=idAgent;
        for action in state.getLegalActions(ghost):
          sucState=state.generateSuccessor(ghost,action)
          if ghost != state.getNumAgents()-1:
            value=min(value,minimax(sucState,depth,ghost+1)[0])
          else:
            value=min(value,minimax(sucState,depth-1,0)[0])
        return (value,None)

    result=minimax(gameState,self.depth,0)
    #print result[0]
    return result[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def minimax(state,depth,idAgent,alpha,beta):
      if (depth == 0) or state.isWin() or state.isLose():
        return (self.evaluationFunction(state),None)
      #max
      if idAgent == 0:
        if alpha >= beta:
          return (float('-inf'),None)
        value=float('-inf')
        actionC=None
        for action in state.getLegalActions(0):
          result=minimax(state.generateSuccessor(0, action),depth-1,1,alpha,beta)
          if result[0] > value:
            value=result[0]
            actionC=action

        if value > alpha:
          alpha=value

        return (value,actionC)
      #min
      else:
        if alpha >= beta:
          return (float('inf'),None)
        value=float('inf')
        ghost=idAgent;
        for action in state.getLegalActions(ghost):
          sucState=state.generateSuccessor(ghost,action)
          if ghost != state.getNumAgents()-1:
            value=min(value,minimax(sucState,depth,ghost+1,alpha,beta)[0])
          else:
            value=min(value,minimax(sucState,depth-1,0,alpha,beta)[0])

        if value < beta:
          beta=value

        return (value,None)

    result=minimax(gameState,self.depth,0,float('-inf'),float('inf'))
    #print result[0]
    return result[1]

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
    "*** YOUR CODE HERE ***"
    def minimax(state,depth,idAgent):
      if (depth == 0) or state.isWin() or state.isLose():
        return (self.evaluationFunction(state),'meh')
      #max
      if idAgent == 0:
        value=float('-inf')
        actionC=None
        for action in state.getLegalActions(0):
          result=minimax(state.generateSuccessor(0, action),depth-1,1)
          if result[0] > value:
            value=result[0]
            actionC=action
        return (value,actionC)
      #min
      else:
        value=float('inf')
        expValue=0.0
        ghost=idAgent;
        for action in state.getLegalActions(ghost):
          sucState=state.generateSuccessor(ghost,action)
          if ghost != state.getNumAgents()-1:
            value=minimax(sucState,depth,ghost+1)
          else:
            value=minimax(sucState,depth-1,0)
          expValue+=(value[0]/len(state.getLegalActions(ghost)))
        return (expValue,None)

    result=minimax(gameState,self.depth,0)
    #print result[0]
    return result[1]

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"

   # Useful information you can extract from a GameState (pacman.py)
  curPos = currentGameState.getPacmanPosition()
  oldFood = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  foodList = oldFood.asList();

  if currentGameState.isWin():
    return float('inf')

  minGhostDist=float('inf')
  ghostDist=0
  for ghost in ghostStates:
    dist=manhattanDistance(curPos,ghost.getPosition())
    minGhostDist=min(minGhostDist,dist)
    ghostDist+=dist

    ghostW=1;
  if minGhostDist < 2:
    ghostW=float('inf')

  minFoodDist=float('inf')
  for food in foodList:
    minFoodDist=min(minFoodDist,dist)

  value=scoreEvaluationFunction(currentGameState)
  value-=minFoodDist*2
  value+=minGhostDist*2
  value/=ghostW

  return value

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

