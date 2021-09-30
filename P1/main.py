import sys
# import fileinput

import numpy as np
import random

from numpy import ndarray

import exceptions
import interpreter


global goal_state
goal_state = np.array([[0,1,2],[3,4,5],[6,7,8]])
global current_state
current_state = np.array([[0,1,2],[3,4,5],[6,7,8]])
global maximum_nodes

"""Note that in this program, printouts of the state  as well as the argument for setting a state use characters to
represent pieces such as '2' for the 2 piece, and especially of note, 'b' for the blank piece.  However, in memory,
the puzzle is represented as a 2-dimensional integer array in which 0 represnts the blank piece, and 1 represents the
1 piece, etc.  It is simpler for the program that way."""

#if __name__ == "__main__":
#    main()

#def main():
#    interpreter.interpreter(sys.argv[1])


def print_state():  #prints the current puzzle state represented by characters
    print("Current State:")
    #print(current_state)
    currentStateReadable = np.array([['b','b','b'],['b','b','b'],['b','b','b']])
    i=0
    while i < current_state[0].size:
        j = 0
        while j < current_state[:, 0].size:
            currentStateReadable[i][j] = int_to_string_representation(current_state[i][j])
            j+=1
        i+=1
    print(currentStateReadable)

def move(direction):
    """This function acts on the current_state variable and either does not move if the move is impossible, or does
        a move depending on the direction string variable: up, down, left, or right."""
    """If the move is completed, it returns 1.  If not, then it returns 0."""
    global current_state
    blankPosition = (0,0)
    i=0
    while i < current_state[0].size:  # this finds the blank piece's position
        j = 0
        while j < current_state[:, 0].size:
            if current_state[i][j] == 0:
                blankPosition = (i,j)
            j+=1
        i+=1

    if direction == 'up':
        if blankPosition[0] == 0:  # this means that a move up is not possible.
            raise exceptions.move_impossible
        else:
            switchValue = current_state[blankPosition[0] - 1][blankPosition[1]]
            current_state[blankPosition[0] - 1][blankPosition[1]] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state
    if direction == 'down':
        if blankPosition[0] == 2:  # this means that a move down is not possible.
            raise exceptions.move_impossible
        else:
            switchValue = current_state[blankPosition[0] + 1][blankPosition[1]]
            current_state[blankPosition[0] + 1][blankPosition[1]] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state
    if direction == 'left':
        if blankPosition[1] == 0:  # this means that a move left is not possible.
            raise exceptions.move_impossible
        else:
            switchValue = current_state[blankPosition[0]][blankPosition[1] - 1]
            current_state[blankPosition[0]][blankPosition[1] - 1] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state
    if direction == 'right':
        if blankPosition[1] == 2:  # this means that a move right is not possible.
            raise exceptions.move_impossible
        else:
            switchValue = current_state[blankPosition[0]][blankPosition[1] + 1]
            current_state[blankPosition[0]][blankPosition[1] + 1] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state

def move_node(direction, node):
    """This function is very similar to the first move() function, but instead of acting on the current_state
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


def int_to_string_representation(integer):
    if integer == 0:
        return 'b'
    else:
        return str(integer)


def string_representation_to_int(string):
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
    current_state[0][0] = string_representation_to_int(stateString[0])
    current_state[0][1] = string_representation_to_int(stateString[1])
    current_state[0][2] = string_representation_to_int(stateString[2])

    current_state[1][0] = string_representation_to_int(stateString[4])
    current_state[1][1] = string_representation_to_int(stateString[5])
    current_state[1][2] = string_representation_to_int(stateString[6])

    current_state[2][0] = string_representation_to_int(stateString[8])
    current_state[2][1] = string_representation_to_int(stateString[9])
    current_state[2][2] = string_representation_to_int(stateString[10])


def randomize_state(n):
    """This takes the goal state, makes n number of random moves from it, and stores it in the current_state variable."""
    current_state = goal_state
    i=0
    while i < n:
        randomValue = random.random() #float in between 0.0 and 1.0
        worked = 0
        if (randomValue >= 0.0) & (randomValue <= 0.25):
            try:
                move('up')
                worked +=1
            except exceptions.move_impossible:
                pass
        elif (randomValue > 0.25) & (randomValue <= 0.50):
            try:
                move('down')
                worked +=1
            except exceptions.move_impossible:
                pass
        elif (randomValue > 0.50) & (randomValue <= 0.75):
            try:
                move('left')
                worked +=1
            except exceptions.move_impossible:
                pass
        elif (randomValue > 0.75) & (randomValue <= 1.0):
            try:
                move('right')
                worked +=1
            except exceptions.move_impossible:
                pass

        # This ensures that n number of actual moves as opposed to attempted moves take place by only
        # incrementing i when the move() function executes successfully:
        if worked != 0:
            i+=1


def max_nodes(n):
    global maximum_nodes
    maximum_nodes = n


def heuristic_one():
    """this function operates on the current state matrix variable and returns the number of misplaced tiles"""
    numMisplaced = 0
    i = 0
    while i < current_state[0].size:
        j = 0
        while j < current_state[:, 0].size:
            if current_state[i][j] != goal_state[i][j]:
                numMisplaced += 1
            j += 1
        i += 1
    return numMisplaced


def heuristic_two():
    """this function operates on the current state matrix variable and returns the sum of the distances of tiles'
    positions in the current state from their positions in the goal state."""

    """goalStateCoordinates is an array containing the coordinates of the tiles in the goal state matrix.  So, for
    example, goalStateCoordinates(4) returns the i,jth coordinates of tile 4 in the goal state: (1,1)."""
    goalStateCoordinates = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    sumOfDistances = 0
    i = 0
    while i < current_state[0].size:
        j = 0
        while j < current_state[:, 0].size:
            #this calculates the Manhattan distance, not the Euclidean distance, because it was unspecified.
            sumOfDistances += ((abs(i - goalStateCoordinates[current_state[i][j]][0])) + (abs(j - goalStateCoordinates[current_state[i][j]][1])))
            j += 1
        i += 1
    return sumOfDistances


class node:
    def __init__(self, state):
        self.parent = None
        self.move = None #the move that took it from its parent to its current state
        self.state = np.copy(state)
        print("new node intialized with state:")
        print(state)
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
    #def getState(self):
    #    return self.state


class priority_queue:
    def __init__(self, heuristic):
        self.heuristic = heuristic
        self.queue = []
        self.size = 0

    def pop(self):
        if(self.size < 1):
            raise exceptions.queue_error
        if (self.heuristic == "h1"):
            maximumIndex = 0
            for i in range(len(self.queue)):
                # comparing evaluation functions:
                if self.queue[i].pathLength() + self.queue[i].state.heuristic_one() > \
                    self.queue[maximumIndex].pathLength() + self.queue[maximumIndex].state.heuristic_one():
                    maximumIndex = i
            returnable = self.queue[maximumIndex]
            del self.queue[maximumIndex]
            return returnable
        elif (self.heuristic == "h2"):
            maximumIndex = 0
            for i in range(len(self.queue)):
                #comparing evaluation functions:
                if self.queue[i].pathLength() + self.queue[i].state.heuristic_two() > \
                    self.queue[maximumIndex].pathLength() + self.queue[maximumIndex].state.heuristic_two():
                    maximumIndex = i
            returnable = self.queue[maximumIndex]
            del self.queue[maximumIndex]
            return returnable


    def insert(self, node):
        self.queue.append(node)
        self.size += 1


def solve_a_star(heuristic):
    initialNode = node()
    initialNode.state = current_state
    frontier = priority_queue(heuristic)
    numNodes = 0
    while priority_queue.size > 0 & numNodes < max_nodes:
        current_node = priority_queue.pop
        numNodes += 1

print_state()
new_node = node(current_state)

move('down')

print_state()
new_node_2 = node(current_state)
new_node_2.parent = new_node
new_node_2.move = 'down'

move('right')

print_state()
new_node_3 = node(current_state)
new_node_3.parent = new_node_2
new_node_3.move = 'right'

print('pause')
print(new_node_2.state)









