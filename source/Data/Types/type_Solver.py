from pysat.formula import CNF
from pysat.solvers import Solver, SolverNames


class Clause(list[int]):
    """A clause in a CNF formula represented by a list of literals."""

    def __init__(self, literals: list[int]):
        """Initialize the clause.

        ### Parameters
        - literals: The list of literals in the clause.
        """
        super().__init__(literals)

    def __str__(self):
        """Return the string representation of the clause."""
        return f"[{', '.join(map(str, self))}]"

    def __list__(self):
        """Return the list representation of the clause."""
        return list(self)


class CNF(CNF):
    """A CNF formula or a conjunction of clauses"""

    def __init__(self, clauses: list[Clause]):
        """Initialize the CNF formula.

        ### Parameters
        @clauses: The list of clauses in the CNF formula.
        """
        super().__init__(from_clauses=clauses)

    def __str__(self):
        """Return the string representation of the CNF formula."""
        return "\n".join(map(str, self.clauses))


class Solver(Solver):
    """A SAT solver.

    ### Attributes
    - name: The name of the solver.
    """

    def __init__(
        self,
        solverName: SolverNames = "glucose3",
        cnf: CNF | list[Clause] | list[list[int]] = [],
    ):
        """Initialize the SAT solver.

        ### Parameters
        - solver: The name of the solver to use.
        - cnf: The CNF formula to solve
        """
        self.name: SolverNames = solverName
        super().__init__(name=self.name, bootstrap_with=cnf or [])

    def addClause(self, clause: Clause, noReturn: bool = True) -> bool | None:
        """Add a clause to the formula."

        ### Parameters"
        - clause: The clause to add to the formula."
        - noReturn: Whether to return the result of the operation.""

        ### Returns
        - The result of the operation if `noReturn` is `False`.
        """
        return super().add_clause(list(clause), no_return=noReturn)

    def getModel(self) -> list[int] | None:
        """Return the model of the formula.

        ### Returns
        - The model of the formula if it exists.
        """
        return super().get_model()

    def solve(self) -> bool:
        """Solve the formula.

        ### Returns
        - Whether the formula is satisfiable.
        """
        return super().solve()

    def __str__(self):
        """Return the string representation of the solver."""
        return f"{self.name} solver"


__all__ = ["Clause", "CNF", "Solver"]
