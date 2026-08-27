"""File containing syntax errors to test AST fault tolerance."""

def valid_function_before(x: int) -> int:
    """This function is valid and defined before the syntax error."""
    return x * 2

                                                          
if True
    x = 100

def valid_function_after(y: str) -> str:
    """This function is valid and defined after the syntax error."""
    return y.upper()
