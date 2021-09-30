class node_error(Exception):
    """This exception is raised when the amount of maxNodes is surpassed"""
    pass


class queue_error(Exception):
    """The priority queue cannot be popped from because it is empty."""
    pass


class move_impossible(Exception):
    """The move is not possible because the blank tile is at a boundary."""
    pass
