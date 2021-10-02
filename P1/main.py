import sys

import numpy as np
import random

from numpy import ndarray

import exceptions

goal_state= np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
current_state = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
maximum_nodes = 0

random.seed(142)  # initializing Python's RNG  #342


def interpreter(filename):
    # filename = "P1-test.txt"
    command_file = open(filename, 'r')
    commands = command_file.readlines()  # creates a list of the individual lines of the .txt file
    # Trying to see if it matched a command in main():
    for command in commands:
        if (len(command) > 1) & (command[0] != "#"):
            command_plus_args = command.split()  # splits the given command into word strings
            if command_plus_args[0] == "move":
                if len(command_plus_args) > 2:
                    raise exceptions.argument_number_error
                try:
                    move(command_plus_args[1])
                except exceptions.move_impossible_error:
                    print("this move was not accepted")
                    pass
            elif command_plus_args[0] == "setState":
                if len(command_plus_args) > 4:
                    raise exceptions.argument_number_error
                state_string = command_plus_args[1] + " " + command_plus_args[2] + " " + command_plus_args[3]
                set_state(state_string)
            elif command_plus_args[0] == "solve":
                if len(command_plus_args) > 3:
                    raise exceptions.argument_number_error
                if command_plus_args[1] == "A-star":
                    heuristic = command_plus_args[2]
                    solve_a_star(heuristic)
                elif command_plus_args[1] == "beam":
                    k = int(command_plus_args[2])
                    solve_beam(k)
                else:
                    raise exceptions.command_error
            elif command_plus_args[0] == "maxNodes":
                if len(command_plus_args) > 2:
                    raise exceptions.argument_number_error
                n = int(command_plus_args[1])
                max_nodes(n)
            elif command_plus_args[0] == "randomizeState":
                if len(command_plus_args) > 2:
                    raise exceptions.argument_number_error
                n = int(command_plus_args[1])
                randomize_state(n)
            elif command_plus_args[0] == "printState":
                if len(command_plus_args) > 2:
                    raise exceptions.argument_number_error
                print_state()
            else:
                raise exceptions.command_error


def print_state():  # prints the current puzzle state represented by characters
    print("Current State:")
    # print(current_state)
    currentStateReadable = np.array([['b', 'b', 'b'], ['b', 'b', 'b'], ['b', 'b', 'b']])
    i = 0
    while i < current_state[0].size:
        j = 0
        while j < current_state[:, 0].size:
            currentStateReadable[i][j] = int_to_string_representation(current_state[i][j])
            j += 1
        i += 1
    print(currentStateReadable)
    print(" ")


def move(direction):
    """This function acts on the current_state variable and either does not move if the move is impossible, or does
        a move depending on the direction string variable: up, down, left, or right."""
    """If the move is completed, it returns 1.  If not, then it returns 0."""
    global current_state
    blankPosition = (0, 0)
    i = 0
    while i < current_state[0].size:  # this finds the blank piece's position
        j = 0
        while j < current_state[:, 0].size:
            if current_state[i][j] == 0:
                blankPosition = (i, j)
            j += 1
        i += 1

    if direction == 'up':
        if blankPosition[0] == 0:  # this means that a move up is not possible.
            raise exceptions.move_impossible_error
        else:
            switchValue = current_state[blankPosition[0] - 1][blankPosition[1]]
            current_state[blankPosition[0] - 1][blankPosition[1]] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state
    if direction == 'down':
        if blankPosition[0] == 2:  # this means that a move down is not possible.
            raise exceptions.move_impossible_error
        else:
            switchValue = current_state[blankPosition[0] + 1][blankPosition[1]]
            current_state[blankPosition[0] + 1][blankPosition[1]] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state
    if direction == 'left':
        if blankPosition[1] == 0:  # this means that a move left is not possible.
            raise exceptions.move_impossible_error
        else:
            switchValue = current_state[blankPosition[0]][blankPosition[1] - 1]
            current_state[blankPosition[0]][blankPosition[1] - 1] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state
    if direction == 'right':
        if blankPosition[1] == 2:  # this means that a move right is not possible.
            raise exceptions.move_impossible_error
        else:
            switchValue = current_state[blankPosition[0]][blankPosition[1] + 1]
            current_state[blankPosition[0]][blankPosition[1] + 1] = 0
            current_state[blankPosition[0]][blankPosition[1]] = switchValue
            return current_state


def move_node(direction, node):
    """This function is extremely similar to the move() function, but instead of acting on the current_state
     variable, it takes the state from the node parameter, acts on it, and either does not move if the move is
     impossible, or completes the move and returns that state."""
    """if it cannot complete the move, it raises an error"""
    node_state = np.copy(node.state)
    blank_position = (0, 0)
    i = 0
    while i < node_state[0].size:  # this finds the blank piece's position
        j = 0
        while j < node_state[:, 0].size:
            if node_state[i][j] == 0:
                blank_position = (i, j)
            j += 1
        i += 1

    if direction == 'up':
        if blank_position[0] == 0:  # this means that a move up is not possible.
            raise exceptions.move_impossible_error
        else:
            switch_value = node_state[blank_position[0] - 1][blank_position[1]]
            node_state[blank_position[0] - 1][blank_position[1]] = 0
            node_state[blank_position[0]][blank_position[1]] = switch_value
            return node_state
    if direction == 'down':
        if blank_position[0] == 2:  # this means that a move down is not possible.
            raise exceptions.move_impossible_error
        else:
            switch_value = node_state[blank_position[0] + 1][blank_position[1]]
            node_state[blank_position[0] + 1][blank_position[1]] = 0
            node_state[blank_position[0]][blank_position[1]] = switch_value
            return node_state
    if direction == 'left':
        if blank_position[1] == 0:  # this means that a move left is not possible.
            raise exceptions.move_impossible_error
        else:
            switch_value = node_state[blank_position[0]][blank_position[1] - 1]
            node_state[blank_position[0]][blank_position[1] - 1] = 0
            node_state[blank_position[0]][blank_position[1]] = switch_value
            return node_state
    if direction == 'right':
        if blank_position[1] == 2:  # this means that a move right is not possible.
            raise exceptions.move_impossible_error
        else:
            switch_value = node_state[blank_position[0]][blank_position[1] + 1]
            node_state[blank_position[0]][blank_position[1] + 1] = 0
            node_state[blank_position[0]][blank_position[1]] = switch_value
            return node_state


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


def set_state(state_string):
    current_state[0][0] = string_representation_to_int(state_string[0])
    current_state[0][1] = string_representation_to_int(state_string[1])
    current_state[0][2] = string_representation_to_int(state_string[2])

    current_state[1][0] = string_representation_to_int(state_string[4])
    current_state[1][1] = string_representation_to_int(state_string[5])
    current_state[1][2] = string_representation_to_int(state_string[6])

    current_state[2][0] = string_representation_to_int(state_string[8])
    current_state[2][1] = string_representation_to_int(state_string[9])
    current_state[2][2] = string_representation_to_int(state_string[10])


def randomize_state(n):
    """This resets the global current_state variable to the goal and then makes n random successful moves on it."""
    global current_state
    current_state = np.copy(goal_state)
    i = 0
    while i < n:
        randomValue = random.random()  # float in between 0.0 and 1.0
        worked = 0
        if (randomValue >= 0.0) & (randomValue <= 0.25):
            try:
                move('up')
                worked += 1
            except exceptions.move_impossible_error:
                pass
        elif (randomValue > 0.25) & (randomValue <= 0.50):
            try:
                move('down')
                worked += 1
            except exceptions.move_impossible_error:
                pass
        elif (randomValue > 0.50) & (randomValue <= 0.75):
            try:
                move('left')
                worked += 1
            except exceptions.move_impossible_error:
                pass
        elif (randomValue > 0.75) & (randomValue <= 1.0):
            try:
                move('right')
                worked += 1
            except exceptions.move_impossible_error:
                pass

        # This ensures that n number of actual moves as opposed to attempted moves take place by only
        # incrementing i when the move() function executes successfully:
        if worked != 0:
            i += 1


def max_nodes(n):
    global maximum_nodes
    maximum_nodes = n


def heuristic_one(node):
    """this function looks at node.state and returns the number of misplaced tiles"""
    cur_state = np.copy(node.state)
    num_misplaced = 0
    i = 0
    while i < cur_state[0].size:
        j = 0
        while j < cur_state[:, 0].size:
            if cur_state[i][j] != goal_state[i][j]:
                num_misplaced += 1
            j += 1
        i += 1
    return num_misplaced


def heuristic_two(node):
    """this function looks at node.state and returns the sum of the distances of tiles'
    positions in the current state from their positions in the goal state."""

    """goal_state_coordinates is an array containing the coordinates of the tiles in the goal state matrix.  So, for
    example, goal_state_coordinates(4) returns the i,jth coordinates of tile 4 in the goal state: (1,1)."""
    goal_state_coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    sum_of_distances = 0

    cur_state = np.copy(node.state)
    i = 0
    while i < cur_state[0].size:
        j = 0
        while j < cur_state[:, 0].size:
            # this calculates the Manhattan distance, not the Euclidean distance, because it was unspecified.
            if(i != 0) & (j != 0):
                sum_of_distances += ((abs(i - goal_state_coordinates[current_state[i][j]][0])) + (
                    abs(j - goal_state_coordinates[current_state[i][j]][1])))
            j += 1
        i += 1
    return sum_of_distances


class node:
    def __init__(self, state):
        self.parent = None
        self.move = None  # the move that took it from its parent to its current state
        self.state = np.copy(state)
        self.path_length = None

    def get_path_length(self):
        if self.parent == None:  # check and see if this is the initial state node
            return 0
        if self.path_length != None:  # check and see if we have run this before
            return self.path_length
        cur_node = self  # trace back and see how many nodes there are until we get to the initial state node
        self.path_length = 0
        while cur_node.parent != None:
            cur_node = cur_node.parent
            self.path_length += 1
        return self.path_length

    # def get_state(self):
    #    return self.state
    # def set_state(self, state):
    #    self.state = state


class a_star_priority_queue:
    def __init__(self, heuristic):
        self.heuristic = heuristic
        self.queue = []
        self.size = 0

    def pop(self):
        if self.size < 1:
            raise exceptions.queue_error
        minimum_index = 0
        for i in range(len(self.queue)):
            # comparing evaluation functions:
            if self.evaluation_function(self.queue[i]) < self.evaluation_function(self.queue[minimum_index]):
                minimum_index = i
        returnable = self.queue[minimum_index]
        del self.queue[minimum_index]
        self.size -= 1
        return returnable

    def insert(self, node):
        self.queue.append(node)
        self.size += 1

    def clear(self):
        self.queue = []
        self.size = 0

    def evaluation_function(self, node):
        if self.heuristic == "h1":
            return node.get_path_length() + heuristic_one(node)
        if self.heuristic == "h2":
            return node.get_path_length() + heuristic_two(node)



class beam_priority_queue:
    def __init__(self):
        self.queue = []
        self.size = 0

    def pop(self):
        if self.size < 1:
            raise exceptions.queue_error
        minimum_index = 0
        for i in range(len(self.queue)):
            # comparing evaluation functions:
            if heuristic_one(self.queue[i]) < heuristic_one(self.queue[minimum_index]):
                minimum_index = i
        returnable = self.queue[minimum_index]
        del self.queue[minimum_index]
        self.size -= 1
        return returnable

    def insert(self, node):
        self.queue.append(node)
        self.size += 1

    def clear(self):
        self.queue = []
        self.size = 0


def check_for_success_a_star(node, frontier):
    if np.array_equal(node.state, goal_state):  # first, checking to see if this is a solution
        # print("found a soln")
        # print(repr(frontier.evaluation_function(node)))
        # if frontier.size > 0:  # checking to see if shorter solutions are possible
            # if node.get_path_length() > frontier.shortest_path():
                # return False
        print("SUCCESS: ")  # SUCCESS prints
        print("Number of moves needed to find solution: " + repr(node.get_path_length()))
        print("Solution: ")
        moves = []
        while node.parent is not None:
            moves.append(node.move)
            node = node.parent
        moves = moves[::-1]  # reversing array
        for cur_move in moves:
            print(cur_move)
        print(" ")
        return True
    else:
        return False


def check_for_success_beam(node):
    if np.array_equal(node.state, goal_state):
        print("SUCCESS: ")  # SUCCESS prints
        print("Number of moves needed to find solution: " + repr(node.get_path_length()))
        print("Solution: ")
        moves = []
        while node.parent is not None:
            moves.append(node.move)
            node = node.parent
        moves = moves[::-1]  # reversing array
        for cur_move in moves:
            print(cur_move)
        print(" ")
        return True
    else:
        return False


def solve_a_star(heuristic):
    root_node = node(current_state)
    frontier = a_star_priority_queue(heuristic)
    frontier.insert(root_node)
    num_nodes = 0
    success = False
    success = check_for_success_a_star(root_node, frontier)
    while (frontier.size > 0) & (num_nodes <= maximum_nodes) & (success == False):
        # pop a node from the frontier:
        cur_node = frontier.pop()

        # expand frontier:
        for moves in ['up', 'down', 'left', 'right']:
            child_node = node(cur_node.state)
            try:
                child_node.state = np.copy(move_node(moves, cur_node))
                child_node.move = moves
                child_node.parent = cur_node
                success = check_for_success_a_star(child_node, frontier)
                if success:
                    return
                frontier.insert(child_node)
                num_nodes += 1
                if num_nodes > maximum_nodes:
                    # raise exceptions.node_error
                    print("A* Failure: Maximum number of nodes surpassed")
                    print(repr(frontier.evaluation_function(frontier.pop())))
                    print(" ")
                    return
            except exceptions.move_impossible_error:
                pass
    if not success:
        print("FAILURE")


def solve_beam(k):
    root_node = node(current_state)
    frontier = beam_priority_queue()
    frontier.insert(root_node)
    success = False
    num_nodes = 0

    success = check_for_success_beam(root_node)  # checking the root node first
    if success:
        return

    while (frontier.size > 0) & (num_nodes <= maximum_nodes) & (success == False):
        # make a new queue and empty out the priority queue:
        temp_queue_1 = []
        k_temp = k
        while (k_temp > 0) & (frontier.size > 0):
            temp_queue_1.append(frontier.pop())
            k_temp -= 1
        children_added = 0
        for cur_node in temp_queue_1:
            children_added = 0
            for moves in ['up', 'down', 'left', 'right']:
                child_node = node(cur_node.state)
                try:
                    child_node.state = np.copy(move_node(moves, cur_node))
                    child_node.move = moves
                    child_node.parent = cur_node
                    success = check_for_success_beam(child_node)
                    if success:
                        return
                    frontier.insert(child_node)
                    num_nodes += 1
                    if num_nodes > maximum_nodes-1:
                        # raise exceptions.node_error
                        print("BEAM FAILURE: Maximum number of nodes surpassed")
                        print(" ")
                        return
                    children_added += 1
                except exceptions.move_impossible_error:
                    pass
        # none of the new children were the solution, now we must add the parents back:
        for cur_node in temp_queue_1:
            frontier.insert(cur_node)
        # and then refine it with children and parents both in:
        k_temp = k
        temp_queue_2 = []
        # gather k of the best entries from frontier
        while (k_temp > 0) & (frontier.size > 0):
            temp_queue_2.append(frontier.pop())
            k_temp -= 1
        # get rid of all of the extra nodes in frontier:
        frontier.clear()
        # insert the k nodes back in
        for cur_node in temp_queue_2:
            frontier.insert(cur_node)


if __name__ == '__main__':
    # interpreter()
    # interpreter("P1-test.txt")
    interpreter("P1_jkm100_test_file.txt")

# max_nodes(1000)
# randomize_state(9)
# print_state()
# solve_beam(8)

