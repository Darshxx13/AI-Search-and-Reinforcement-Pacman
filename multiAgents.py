# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self):
        self.previousPositions = []


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) 

        # see pac's prev pos
        pacmanPos = gameState.getPacmanPosition()
        self.previousPositions.append(pacmanPos)
        if len(self.previousPositions) > 5:  # last 10 pos
            self.previousPositions.pop(0)

        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            if newScaredTimes[i] > 0:  #EATTTT
                score += 10 / (ghostDist + 1)
            elif ghostDist <= 1:  # runnsies
                score -= 1000
            elif ghostDist <= 2:  #dangerss
                score -= 50  

        # praying n crying this works
        if newFood:
            foodDistance = self.bfsFindFood(currentGameState, newPos)
            if foodDistance is not None:
                score += 20 / (foodDistance + 1)  

        # loops
        if newPos in self.previousPositions:
            score -= 100 

        if action == Directions.STOP:
            score -= 50  

        return score
    
    def bfsFindFood(self, gameState, startPosition):
        walls = gameState.getWalls()
        queue = util.Queue()  
        queue.push((startPosition, 0))  
        visited = set()

        while not queue.isEmpty():
            position, distance = queue.pop()

            if position in visited:
                continue
            visited.add(position)

            if position in gameState.getFood().asList():
                return distance

            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(direction)
                nextX, nextY = int(position[0] + dx), int(position[1] + dy)

                if not walls[nextX][nextY]:  
                    queue.push(((nextX, nextY), distance + 1))

        return None


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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
       
        def minimax(state, depth, agentIndex):
            
            # win, lose, max depth
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # pac
            if agentIndex == 0:
                return maxValue(state, depth)

            # ghost
            else:
                return minValue(state, depth, agentIndex)

        def maxValue(state, depth): # paccie
            
            legalActions = state.getLegalActions(0) 
            if not legalActions:
                return self.evaluationFunction(state) #moves available check

            maxbestValue = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                maxbestValue = max(maxbestValue, minimax(successor, depth, 1)) 
            return maxbestValue

        def minValue(state, depth, agentIndex): # ghosts
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state) 
            minValue = float('inf')
            numAgents = state.getNumAgents()
            nextAgent = agentIndex + 1

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                # R U THE PROBLEM
                if nextAgent == numAgents:
                    minValue = min(minValue, minimax(successor, depth + 1, 0))  
                else:
                    minValue = min(minValue, minimax(successor, depth, nextAgent))  
            return minValue

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        bestAction = None
        bestScore = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1)  # da first ghost
            if value > bestScore:
                bestScore = value
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-inf')  
        beta = float('inf')  
        bestAction = None
        bestScore = float('-inf')

        legalActions = gameState.getLegalActions(0)

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.pruning(successor, 1, 0, alpha, beta) # start at 1st ghost
            
            if value > bestScore:
                bestScore = value
                bestAction = action

            alpha = max(alpha, bestScore) 

        return bestAction

    def pruning(self, state, agentIndex, depth, alpha, beta):
        # win, lose, max depth
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return self.maxValue(state, alpha, beta, depth)

        else:
            return self.minValue(state, alpha, beta, depth, agentIndex)

    def maxValue(self, state, alpha, beta, depth):
        v = float('-inf')
        legalActions = state.getLegalActions(0) 

        if not legalActions:
            return self.evaluationFunction(state) 

        for action in legalActions:
            successor = state.generateSuccessor(0, action)
            v = max(v, self.pruning(successor, 1, depth, alpha, beta))

            if v > beta: # prune check
                return v  
            alpha = max(alpha, v)
        return v

    def minValue(self, state, alpha, beta, depth, agentIndex):
        v = float('inf')
        legalActions = state.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(state)

        nextAgent = agentIndex + 1 #next ghost

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)

            if nextAgent == state.getNumAgents():  # check last ghost
                v = min(v, self.pruning(successor, 0, depth + 1, alpha, beta))
            else:
                v = min(v, self.pruning(successor, nextAgent, depth, alpha, beta))

            if v < alpha:
                return v  
            beta = min(beta, v)  
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        bestAction = None
        bestScore = float('-inf')
        legalActions = gameState.getLegalActions(0) 

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.expectimax(successor, 1, 0) 
            
            if value > bestScore:
                bestScore = value
                bestAction = action 

        return bestAction

    def expectimax(self, state, agentIndex, depth):
        # win, lose, max depth
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return self.maxValue(state, depth)

        else:
            return self.expectValue(state, agentIndex, depth)

    def maxValue(self, state, depth):
      
        v = float('-inf')
        legalActions = state.getLegalActions(0) 

        if not legalActions:
            return self.evaluationFunction(state)

        for action in legalActions:
            successor = state.generateSuccessor(0, action)
            v = max(v, self.expectimax(successor, 1, depth)) 

        return v

    def expectValue(self, state, agentIndex, depth):
        
        legalActions = state.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(state)

        numofActions = len(legalActions)
        expectedValue = 0

        nextAgent = agentIndex + 1 if agentIndex < state.getNumAgents() - 1 else 0
        nextDepth = depth + 1 if nextAgent == 0 else depth

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            expectedValue += self.expectimax(successor, nextAgent, nextDepth) / numofActions  # AVG: 1/n

        return expectedValue

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This function optimizes pacman strategy by balancing food collection, ghost avoidance/hunting, capsule usage, and movement efficiency.
    The key feautures are:
    1. Food Priority: Encourages eating the nearest food and penalizes remaining food.
    2. Ghost Awareness: Chases scared ghosts and avoids active ghosts with penalties.
    3. Capsule: Encourages collecting power pellets.
    4: Anti-Camping: Penalizes standing still at one place.
    """
    """
    This evaluation function considers:
    - The current game score.
    - Distance to the nearest food (encourages eating food quickly).
    - Distance to ghosts (avoids active ghosts, chases scared ghosts).
    - Presence of power pellets.
    - Remaining food count (encourages efficient food collection).
    """

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    capsules = currentGameState.getCapsules() 
    score = currentGameState.getScore() 

    # Closer food means higher score
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 10 / (minFoodDist + 1)  
    
    score -= 5 * len(foodList)  

    # My EXTREME Ghost Hunting 
    for i, ghost in enumerate(ghostStates):
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)

        if scaredTimes[i] > 0:  
            score += 100 / (ghostDist + 1)  # Eat the hungry ghost first
        elif ghostDist < 2:  
            score -= 200  # dont get too close to ghost
            #This is wrong!!!!!!
        elif ghostDist < 4:  # Getting closer to ghost
            score -= 50  

    # check his last pos and penalize for camping
    if not hasattr(betterEvaluationFunction, "lastPacmanPos"):
        betterEvaluationFunction.lastPacmanPos = None

    if betterEvaluationFunction.lastPacmanPos == pacmanPos:
        score -= 30 
    betterEvaluationFunction.lastPacmanPos = pacmanPos 

    if capsules:
        minCapsuleDist = min(manhattanDistance(pacmanPos, cap) for cap in capsules)
        score += 50 / (minCapsuleDist + 1)  # eat the capsules more
    return score

# Abbreviation
better = betterEvaluationFunction
