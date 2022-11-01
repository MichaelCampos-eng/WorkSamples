# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from re import S
from sre_constants import BRANCH
from tokenize import String
from typing import Tuple
from webbrowser import get
from xmlrpc.client import Boolean
from game import Directions
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class CuteTree:
    def __init__(self, state : list, parent) -> None:
        self.parent = parent
        self.node = state

    def setBranches(self, children) -> None:
        branches = []
        for child in children:
            branches.append(CuteTree(child, self))
        self.branches = branches

    def getBranches(self):
        return self.branches

    def isGoalState(self, problem : SearchProblem) -> Boolean:
        return problem.isGoalState(self.getState())

    def getNodeChildren(self, problem : SearchProblem):
        return problem.getSuccessors(self.getState())

    def getDirection(self):
        return self.node[1]

    def getState(self):
        return self.node[0]
    
    def getEdgeCost(self):
        if self.node[2] == None:
            return 0
        return self.node[2]
    
    def getCostSoFar(self) -> int:
        return self.distance

    def setCostSoFar(self, distance) -> None:
        self.distance = distance
    
def getPath(node : CuteTree):
        if node.getDirection() == None:
            return []
        return [node.getDirection()] + getPath(node.parent)        

def updateList(children : list, storage : list):
    for child in children:
        storage.push(child)
    return storage 

def updatePQ(children : list, storage : util.PriorityQueue, costSoFar : int):
    for child in children:
        child.setCostSoFar(child.getEdgeCost() + costSoFar)
        storage.push(child, child.getCostSoFar())
    return storage

def updateFringeHeuristic(children : list, storage : util.PriorityQueue, costSoFar : int, heuristic, problem):
    for child in children:
        child.setCostSoFar(child.getEdgeCost() + costSoFar)
        storage.push(child, child.getCostSoFar() + heuristic(child.getState(), problem))
    return storage

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    visited = []
    node = CuteTree([problem.getStartState(), None, None, None], None)
    fringe = updateList([node], fringe)

    while not fringe.isEmpty():
        node = fringe.pop()
        if node.getState() in visited:
            continue
        visited.append(node.getState())
        if node.isGoalState(problem):
            path = getPath(node)
            path.reverse()
            return path
        children = node.getNodeChildren(problem)
        if children == []:
            continue
        node.setBranches(children)
        fringe = updateList(node.getBranches(), fringe)
    return []  

def getStartNode(problem: SearchProblem):
    return CuteTree([problem.getStartState(), None, None], None)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    visited = []
    node = getStartNode(problem)
    fringe = updateList([node], fringe)
    
    while not fringe.isEmpty():
        node = fringe.pop()
        if node.getState() in visited:
            continue
        visited.append(node.getState())
        if node.isGoalState(problem):
            path = getPath(node)
            path.reverse()
            return path
        children = node.getNodeChildren(problem)
        if children == []:
            continue
        node.setBranches(children)
        fringe = updateList(node.getBranches(), fringe)
    return []  

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = []
    fringe = util.PriorityQueue()
    node = CuteTree([problem.getStartState(), None, None], None)
    fringe = updatePQ([node], fringe, 0)

    while not fringe.isEmpty():
        node = fringe.pop()
        if node.getState() in visited:
            continue
        visited.append(node.getState())
        if node.isGoalState(problem):
            path = getPath(node)
            path.reverse()
            return path
        children = node.getNodeChildren(problem)
        if children == []:
            continue
        node.setBranches(children)
        fringe = updatePQ(node.getBranches(), fringe, node.getCostSoFar())
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visited = []
    fringe = util.PriorityQueue()
    node = getStartNode(problem)
    fringe = updateFringeHeuristic([node], fringe, 0, heuristic, problem)

    while not fringe.isEmpty():
        node = fringe.pop()
        if node.getState() in visited:
            continue
        visited.append(node.getState())
        if node.isGoalState(problem):
            path = getPath(node)
            path.reverse()
            return path
        children = node.getNodeChildren(problem)
        if children == []:
            continue
        node.setBranches(children)
        fringe = updateFringeHeuristic(node.getBranches(), fringe, node.getCostSoFar(), heuristic, problem)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch