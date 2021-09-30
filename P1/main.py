import numpy as np
import random

import exceptions



goalState = np.array([[0,1,2],[3,4,5],[6,7,8]])
currentState = np.array([[0,1,2],[3,4,5],[6,7,8]])
maxNodes = 1

#Note that in this program, printouts of the state  as well as the argument for setting a state use characters to
#represent pieces such as '2' for the 2 piece, and especially of note, 'b' for the blank piece.  However, in memory,
#the puzzle is represented as a 2-dimensional integer array in which 0 represnts the blank piece, and 1 represents the
#1 piece, etc.  It is simpler for the program that way.

def printState(): #prints the current puzzle state represented by characters
    print("Current State:")
    #print(currentState)
    currentStateReadable = np.array([['b','b','b'],['b','b','b'],['b','b','b']])
    i=0
    while i < currentState[0].size:
        j = 0
        while j < currentState[:,0].size:
            currentStateReadable[i][j] = intToStringRepresentation(currentState[i][j])
            j+=1
        i+=1
    print(currentStateReadable)

def move(direction):
    """This function acts on the currentState variable and either does not move if the move is impossible, or does
        a move depending on the direction string variable: up, down, left, or right."""
    """If the move is completed, it returns the currentState.  If not, then it returns None."""
    blankPosition = (0,0)
    i=0
    while i < currentState[0].size: # this finds the blank piece's position
        j = 0
        while j < currentState[:,0].size:
            if currentState[i][j] == 0:
                blankPosition = (i,j)
            j+=1
        i+=1

    if direction == 'up':
        if blankPosition[0] == 0: # this means that a move up is not possible.
            return None
        else:
            switchValue = currentState[blankPosition[0]-1][blankPosition[1]]
            currentState[blankPosition[0] - 1][blankPosition[1]] = 0
            currentState[blankPosition[0]][blankPosition[1]] = switchValue
            return currentState
    if direction == 'down':
        if blankPosition[0] == 2:# this means that a move down is not possible.
            return None
        else:
            switchValue = currentState[blankPosition[0] + 1][blankPosition[1]]
            currentState[blankPosition[0] + 1][blankPosition[1]] = 0
            currentState[blankPosition[0]][blankPosition[1]] = switchValue
            return currentState
    if direction == 'left':
        if blankPosition[1] == 0:# this means that a move left is not possible.
            return None
        else:
            switchValue = currentState[blankPosition[0]][blankPosition[1] - 1]
            currentState[blankPosition[0]][blankPosition[1] - 1] = 0
            currentState[blankPosition[0]][blankPosition[1]] = switchValue
            return currentState
    if direction == 'right':
        if blankPosition[1] == 2:# this means that a move right is not possible.
            return None
        else:
            switchValue = currentState[blankPosition[0]][blankPosition[1] + 1]
            currentState[blankPosition[0]][blankPosition[1] + 1] = 0
            currentState[blankPosition[0]][blankPosition[1]] = switchValue
            return currentState

def move(direction, node):
    """This function is very similar to the first move() function, but instead of acting on the currentState
     variable, takes the state from a particular node, acts on it, and either does not move if the move is
     impossible, or completes the move and returns that state."""
    """if it cannot complete the move, it returns None"""
    nodeState = node.state
    blankPosition = (0,0)
    i=0
    while i < nodeState[0].size: # this finds the blank piece's position
        j = 0
        while j < nodeState[:,0].size:
            if nodeState[i][j] == 0:
                blankPosition = (i,j)
            j+=1
        i+=1

    if direction == 'up':
        if blankPosition[0] == 0: # this means that a move up is not possible.
            return None
        else:
            switchValue = nodeState[blankPosition[0]-1][blankPosition[1]]
            nodeState[blankPosition[0] - 1][blankPosition[1]] = 0
            nodeState[blankPosition[0]][blankPosition[1]] = switchValue
            return nodeState
    if direction == 'down':
        if blankPosition[0] == 2:# this means that a move down is not possible.
            return None
        else:
            switchValue = nodeState[blankPosition[0] + 1][blankPosition[1]]
            nodeState[blankPosition[0] + 1][blankPosition[1]] = 0
            nodeState[blankPosition[0]][blankPosition[1]] = switchValue
            return nodeState
    if direction == 'left':
        if blankPosition[1] == 0:# this means that a move left is not possible.
            return None
        else:
            switchValue = nodeState[blankPosition[0]][blankPosition[1] - 1]
            nodeState[blankPosition[0]][blankPosition[1] - 1] = 0
            nodeState[blankPosition[0]][blankPosition[1]] = switchValue
            return nodeState
    if direction == 'right':
        if blankPosition[1] == 2:# this means that a move right is not possible.
            return None
        else:
            switchValue = nodeState[blankPosition[0]][blankPosition[1] + 1]
            nodeState[blankPosition[0]][blankPosition[1] + 1] = 0
            nodeState[blankPosition[0]][blankPosition[1]] = switchValue
            return nodeState

def intToStringRepresentation(integer):
    if integer == 0:
        return 'b'
    else:
        return str(integer)

def StringRepresentationToInt(string):
    if string == 'b':
        return 0
    else:
        if string == '1':
            return 1
        if string == '2':
            return 2
        if string == '3':
            return 3
        if string == '4':
            return 4
        if string == '5':
            return 5
        if string == '6':
            return 6
        if string == '7':
            return 7
        if string == '8':
            return 8

def setState(stateString):
    currentState[0][0] = StringRepresentationToInt(stateString[0])
    currentState[0][1] = StringRepresentationToInt(stateString[1])
    currentState[0][2] = StringRepresentationToInt(stateString[2])

    currentState[1][0] = StringRepresentationToInt(stateString[4])
    currentState[1][1] = StringRepresentationToInt(stateString[5])
    currentState[1][2] = StringRepresentationToInt(stateString[6])

    currentState[2][0] = StringRepresentationToInt(stateString[8])
    currentState[2][1] = StringRepresentationToInt(stateString[9])
    currentState[2][2] = StringRepresentationToInt(stateString[10])

def randomizeState(n):
    """This takes the goal state, makes n number of random moves from it, and stores it in the currentState variable."""
    currentState = goalState
    i=0
    while i < n:
        randomValue = random.random() #float in between 0.0 and 1.0
        worked = 0
        if (randomValue >= 0.0) & (randomValue <= 0.25):
            worked = move('up')
        elif (randomValue > 0.25) & (randomValue <= 0.50):
            worked = move('down')
        elif (randomValue > 0.50) & (randomValue <= 0.75):
            worked = move('right')
        elif (randomValue > 0.75) & (randomValue <= 1.0):
            worked = move('left')

        # This ensures that n number of actual moves as opposed to attempted moves take place by only
        # incrementing i when the move() function executed successfully:
        if worked != None:
            i+=1

def maxNodes(n):
    maxNodes = n

def heuristicOne():
    """this function operates on the current state matrix variable and returns the number of misplaced tiles"""
    numMisplaced = 0
    i = 0
    while i < currentState[0].size:
        j = 0
        while j < currentState[:, 0].size:
            if currentState[i][j] != goalState[i][j]:
                numMisplaced += 1
            j += 1
        i += 1
    return numMisplaced

def heuristicTwo():
    """this function operates on the current state matrix variable and returns the sum of the distances of tiles'
    positions in the current state from their positions in the goal state."""

    """goalStateCoordinates is an array containing the coordinates of the tiles in the goal state matrix.  So, for
    example, goalStateCoordinates(4) returns the i,jth coordinates of tile 4 in the goal state: (1,1)."""
    goalStateCoordinates = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    sumOfDistances = 0
    i = 0
    while i < currentState[0].size:
        j = 0
        while j < currentState[:, 0].size:
            #this calculates the Manhattan distance, not the Euclidean distance, because it was unspecified.
            sumOfDistances += ((abs(i-goalStateCoordinates[currentState[i][j]][0])) + (abs(j-goalStateCoordinates[currentState[i][j]][1])))
            j += 1
        i += 1
    return sumOfDistances

class node():
    def __init__(self, state):
        self.parent = None
        self.move = None #the move that took it from its parent to its current state
        self.state = state
        self.pathLength = None
    def getPathLength(self):
        if self.parent == None: #check and see if this is the initial state node
            return 0
        if self.pathLength != None: #check and see if we have run this before
            return self.pathLength
        currentNode = self #trace back and see how many nodes there are until we get to the initial state node
        self.pathLength = 0
        while currentNode.parent != None:
            currentNode=currentNode.parent
            self.pathLength += 1
        return self.pathLength

def solveAStar(heuristic):
    initialNode = node()
    initialNode.state = currentState
    return AStar(initialNode)

class priorityQueue():
    def __init__(self, heuristic):
        self.heuristic = heuristic
        self.queue = []

    def pop(self):
        if (self.heuristic == "h1"):
            maximumIndex = 0
            for i in range(len(self.queue)):
                if self.queue[i].pathLength() + self.queue[i].state.heuristicOne() > \
                    self.queue[maximumIndex].pathLength() + self.queue[maximumIndex].state.heuristicOne():
                    maximumIndex = i
            returnable = self.queue[maximumIndex]
            del self.queue[maximumIndex]
            return returnable
        elif (self.heuristic == "h2"):
            maximumIndex = 0
            for i in range(len(self.queue)):
                if self.queue[i].pathLength() + self.queue[i].state.heuristicTwo() > \
                    self.queue[maximumIndex].pathLength() + self.queue[maximumIndex].state.heuristicTwo():
                    maximumIndex = i
            returnable = self.queue[maximumIndex]
            del self.queue[maximumIndex]
            return returnable


    def insert(self, node):
        self.queue.append(node)

#def AStar(initialNode):




setState('123 4b5 678')
#printState()
randomizeState(16)
#printState()
newNode = node(currentState)
print(newNode.state)
move('up')
newNode2 = node(currentState)
newNode2.move = 'up'
newNode2.parent = newNode
print(newNode2.getPathLength())

