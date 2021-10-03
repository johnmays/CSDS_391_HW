import main
import matplotlib.pyplot as plt

def experiment_a():
    limits = [5, 10, 50, 100, 500]
    for limit in limits:
        main.max_nodes(limit)
        solved_puzzles = 0
        i = 0
        while i < 100:
            main.randomize_state(25)
            success = main.solve_a_star("h1")
            if success:
                solved_puzzles += 1
            i += 1
        print("Limit of " + repr(limit) + " produced " + repr(solved_puzzles) + "% success rate.")

def experiment_b():
    main.max_nodes(50)
    i = 0
    h1_solved_puzzles = 0
    h2_solved_puzzles = 0
    while i < 500:
        main.randomize_state(25)
        h1_success = main.solve_a_star("h1")
        if h1_success:
            h1_solved_puzzles += 1
        h2_success = main.solve_a_star("h2")
        if h2_success:
            h2_solved_puzzles += 1
        i+=1
    print("A* with h1 solved " + repr(h1_solved_puzzles / 5) + "% of the puzzles.")
    print("A* with h2 solved " + repr(h2_solved_puzzles / 5) + "% of the puzzles.")

def experiment_c():
    main.max_nodes(50)
    i = 0
    average_search_length_h1 = 0
    average_search_length_h2 = 0
    average_search_length_beam = 0
    i = 0
    while i < 500:
        i+=1
    print("")

def experiment_d():
    node_limits = [20, 25, 70, 100]
    scramble_lengths = [7, 9, 12, 14, 30]

    h1_solved_puzzles = 0
    h2_solved_puzzles = 0
    beam_solved_puzzles = 0

    for limit in node_limits:
        for scramble_length in scramble_lengths:
            main.max_nodes(limit)
            i = 0
            while i < 50:
                main.randomize_state(scramble_length)
                h1_success = main.solve_a_star("h1")
                if h1_success:
                    h1_solved_puzzles += 1
                h2_success = main.solve_a_star("h2")
                if h2_success:
                    h2_solved_puzzles += 1
                beam_success = main.solve_beam(11)
                if beam_success:
                    beam_solved_puzzles += 1
                i += 1
    print("A* with h1 solved " + repr(h1_solved_puzzles/10) + "% of problems.")
    print("A* with h2 solved " + repr(h2_solved_puzzles / 10) + "% of problems.")
    print("Beam solved " + repr(h1_solved_puzzles / 10) + "% of problems.")
    print(" ")
    print(h1_solved_puzzles)


def plots():
    x_1 = [10, 50, 100, 500, 1000]
    y_1 = [21, 40, 59, 81, 81]
    x_2 = [5, 10, 50, 100, 500]
    y_2 = [9, 14, 49, 58, 77]
    plt.xlabel('Maximum Number of Nodes')
    plt.ylabel('Percent of Puzzles Solved')
    plt.scatter(x_1, y_1, c = 'b')
    plt.scatter(x_2, y_2, c = 'r')
    plt.show()

