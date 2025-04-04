import Data.data_Types as _Types

from itertools import product
from pysat.card import CardEnc

from Utilities.DefineBridgeVariables import DefineBridgeVariables


def GenerateCNF(level: _Types.Level, edges: list[_Types.Edge]) -> _Types.CNF:
    """Generate the CNF formula for the level.

    ### Parameters:
    - level: The level to generate the CNF formula for.
    - edges: The edges of the level.

    ### Returns:
    - The CNF formula for the level
    """
    # Get the islands
    islands: dict[_Types.Position, int] = level.islands

    # Define the bridge variables and initialize the clause list
    singleBridgeVars, doubleBridgeVars, varCounter = DefineBridgeVariables(edges)
    clauses: list[_Types.Clause] = []

    # Constraint 1: A bridge  cannot be both single and double
    # (¬isSingleBridge_edge ∨ ¬isDoubleBridge_edge)
    for edge in edges:
        isSingleBridge, isDoubleBridge = singleBridgeVars[edge], doubleBridgeVars[edge]
        clauses.append([-isSingleBridge, -isDoubleBridge])

    # Constraint 2: Island bridge count constraints
    islandEdges = {}

    for edge in edges:
        # Get the starting and ending islands of the edge
        island1, island2 = edge.startingPosition, edge.endingPosition

        # Map each island to its connected edges
        islandEdges.setdefault(island1, []).append(edge)
        islandEdges.setdefault(island2, []).append(edge)

    for island, requiredBridgeNumber in islands.items():
        # Get the edges connected to this island
        connectedEdges = islandEdges.get(island, [])

        # Create variables that represent the contribution of each edge to this island
        bridgeVars = []

        for edge in connectedEdges:
            # Single bridge contributes 1
            singleBridgeVar = singleBridgeVars[edge]
            bridgeVars.append(singleBridgeVar)

            # Double bridge contributes 2 (add the variable twice)
            doubleBridgeVar = doubleBridgeVars[edge]
            bridgeVars.append(doubleBridgeVar)
            bridgeVars.append(doubleBridgeVar)

        # Now add the cardinality constraint: exactly requiredBridges bridges
        if bridgeVars:
            # Use AtMostK and AtLeastK to create exactly-K
            atLeast = CardEnc.atleast(
                lits=bridgeVars,
                bound=requiredBridgeNumber,
                encoding=1,
                top_id=varCounter,
            )
            varCounter = atLeast.nv + 1
            clauses.extend(atLeast.clauses)

            atMost = CardEnc.atmost(
                lits=bridgeVars,
                bound=requiredBridgeNumber,
                encoding=1,
                top_id=varCounter,
            )
            varCounter = atMost.nv + 1
            clauses.extend(atMost.clauses)

    # Constraint 3: No crossing bridges
    for edge1, edge2 in product(edges, repeat=2):
        # Avoid duplicate and self-cheking
        if edge1.id >= edge2.id:
            continue

        # Get the starting and ending positions of the edges and their bridge variables
        startRow1, startCol1, endRow1, endCol1 = (
            edge1.startingPosition.y,
            edge1.startingPosition.x,
            edge1.endingPosition.y,
            edge1.endingPosition.x,
        )
        singleBridgeVar1, doubleBridgeVar1 = (
            singleBridgeVars[edge1],
            doubleBridgeVars[edge1],
        )

        startRow2, startCol2, endRow2, endCol2 = (
            edge2.startingPosition.y,
            edge2.startingPosition.x,
            edge2.endingPosition.y,
            edge2.endingPosition.x,
        )
        singleBridgeVar2, doubleBridgeVar2 = (
            singleBridgeVars[edge2],
            doubleBridgeVars[edge2],
        )

        # Check if bridges would cross

        # edge1 is horizontal, edge2 is vertical
        if (
            startRow1 == endRow1
            and startCol2 == endCol2
            and min(startCol1, endCol1) < startCol2 < max(startCol1, endCol1)
            and min(startRow2, endRow2) < startRow1 < max(startRow2, endRow2)
        ):
            clauses.extend(
                [
                    [-singleBridgeVar1, -singleBridgeVar2],
                    [-singleBridgeVar1, -doubleBridgeVar2],
                    [-doubleBridgeVar1, -singleBridgeVar2],
                    [-doubleBridgeVar1, -doubleBridgeVar2],
                ]
            )

        # edge1 is vertical, edge2 is horizontal
        elif (
            startCol1 == endCol1
            and startRow2 == endRow2
            and min(startRow1, endRow1) < startRow2 < max(startRow1, endRow1)
            and min(startCol2, endCol2) < startCol1 < max(startCol2, endCol2)
        ):
            clauses.extend(
                [
                    [-singleBridgeVar1, -singleBridgeVar2],
                    [-singleBridgeVar1, -doubleBridgeVar2],
                    [-doubleBridgeVar1, -singleBridgeVar2],
                    [-doubleBridgeVar1, -doubleBridgeVar2],
                ]
            )

    # Constraint 4: All islands must be connected
    # We'll use a simpler approach - make sure at least |islands|-1 bridges exist
    bridgeVars = []
    for edge in edges:
        # A bridge exists on this edge if either p or q is true
        bridgeExists = varCounter
        singleBridgeVar, doubleBridgeVar = (
            singleBridgeVars[edge],
            doubleBridgeVars[edge],
        )

        # singleBridgeVar → bridgeExists, doubleBridgeVar → bridgeExists
        clauses.append([-singleBridgeVar, bridgeExists])
        clauses.append([-doubleBridgeVar, bridgeExists])

        # ¬bridgeExists → ¬singleBridgeVar ∧ ¬doubleBridgeVar
        clauses.append([-bridgeExists, singleBridgeVar, doubleBridgeVar])

        # Add the bridge variable to the list
        bridgeVars.append(bridgeExists)
        varCounter += 1

    # Need at least |islands|-1 bridges to connect all islands
    if len(islands) > 1:
        atLeast = CardEnc.atleast(
            lits=bridgeVars, bound=len(islands) - 1, encoding=1, top_id=varCounter
        )

        varCounter = atLeast.nv + 1
        clauses.extend(atLeast.clauses)

    return _Types.CNF(clauses)
