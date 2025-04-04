from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton

class SolverWidget(QWidget):
    """Widget for solver selection and controls."""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        
        # Solver selection
        layout.addWidget(QLabel("Solver:"))
        
        self.solverCombo = QComboBox()
        self.solverCombo.addItem("PySAT")
        
        self.solverCombo.addItem("AStar")
        self.solverCombo.addItem("Backtracking")
        self.solverCombo.addItem("BruteForce")
        layout.addWidget(self.solverCombo)
        
        # Solve button
        self.solveButton = QPushButton("Solve")
        layout.addWidget(self.solveButton)
        
        self.setLayout(layout)
        
    def getSelectedSolver(self) -> str:
        """Get the currently selected solver."""
        return self.solverCombo.currentText()