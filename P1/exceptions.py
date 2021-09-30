class NodeError(Exception):
    """This exception is raised when the amount of maxNodes is surpassed"""
    pass

class queueError(Exception):
    """The priority queue cannot be popped from because it is empty."""
    pass