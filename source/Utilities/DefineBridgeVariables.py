import Data.data_Types as _Types


def DefineBridgeVariables(
    edges: list[_Types.Edge],
) -> tuple[dict[_Types.Edge, int], dict[_Types.Edge, int], int]:
    """Define the bridge variables.

    ### Parameters:
    - edges: The edges of the level.

    ### Returns:
    - The single bridge variables, double bridge variables, and the variable counter.
    """
    # Initialize the single and double bridge variables
    singleBridgeVars: dict[_Types.Edge, int] = {}
    doubleBridgeVars: dict[_Types.Edge, int] = {}

    variableCounter = 1

    # Define the bridge variables with each edge
    for edge in edges:
        singleBridgeVars[edge] = variableCounter  # Single bridge variable
        variableCounter += 1
        doubleBridgeVars[edge] = variableCounter  # Double bridge variable
        variableCounter += 1

    return singleBridgeVars, doubleBridgeVars, variableCounter


__all__ = ["DefineBridgeVariables"]
