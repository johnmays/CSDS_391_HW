class node_error(Exception):
    """This exception is raised when the amount of maxNodes is surpassed"""
    pass


class queue_error(Exception):
    """The priority queue cannot be popped from because it is empty."""
    pass


class move_impossible_error(Exception):
    """The move is not possible because the blank tile is at a boundary."""
    pass

class argument_number_error(Exception):
    """Did not expect this many arguments for this command"""
    pass

class command_error(Exception):
    """unexpected command"""
    pass
