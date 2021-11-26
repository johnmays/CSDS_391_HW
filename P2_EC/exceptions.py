class unexpected_class_error(Exception):
    """This exception is raised when a class (virgincia, versicolor, or setosa) was incorrectly passed to a function."""
    pass


class insufficient_arguments_error(Exception):
    """This exception is raised when a function or class was not
    passed the proper # or type of args/kwargs"""
    pass
