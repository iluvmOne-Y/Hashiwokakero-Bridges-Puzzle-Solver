import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QComboBox, QFileDialog,
    QMessageBox, QStatusBar
)
from PyQt5.QtCore import Qt

import Data.data_Types as _Types
import Controllers.FileController as FileController
import Controllers.SolverController as SolverController

from GUI.BoardWidget import BoardWidget
from GUI.SolverWidget import SolverWidget

class MainWindow(QMainWindow):
    """Main window for the Hashiwokakero puzzle solver GUI."""
    
    def __init__(self):
        super().__init__()
        self.level = None
        self.solution = None
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("Hashiwokakero Puzzle Solver")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)
        
        # Create top controls
        topLayout = QHBoxLayout()
        
        # File operations
        self.loadButton = QPushButton("Load Puzzle")
        self.loadButton.clicked.connect(self.loadPuzzle)
        self.saveButton = QPushButton("Save Solution")
        self.saveButton.clicked.connect(self.saveSolution)
        self.saveButton.setEnabled(False)
        
        topLayout.addWidget(self.loadButton)
        topLayout.addWidget(self.saveButton)
        
        # Solver widget
        self.solverWidget = SolverWidget()
        self.solverWidget.solveButton.clicked.connect(self.solvePuzzle)
        topLayout.addWidget(self.solverWidget)
        
        mainLayout.addLayout(topLayout)
        
        # Board widget for displaying the puzzle
        self.boardWidget = BoardWidget()
        mainLayout.addWidget(self.boardWidget)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def loadPuzzle(self):
        """Load a puzzle from a file."""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Puzzle", "source/Inputs", "Text Files (*.txt)"
        )
        
        if fileName:
            try:
                # Extract level number from filename
                import re
                match = re.search(r'input-(\d+)', fileName)
                if match:
                    level_num = int(match.group(1))
                    self.level = FileController.ParseLevelFromFile(level_num)
                    
                    if self.level:
                        self.boardWidget.setLevel(self.level)
                        self.statusBar.showMessage(f"Loaded puzzle level {level_num}")
                        self.solverWidget.setEnabled(True)
                        self.saveButton.setEnabled(False)
                        self.solution = None
                else:
                    raise ValueError("Invalid filename format")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading puzzle: {str(e)}")
                
    def solvePuzzle(self):
        """Solve the current puzzle."""
        if not self.level:
            QMessageBox.warning(self, "Warning", "Please load a puzzle first.")
            return
            
        solver = self.solverWidget.getSelectedSolver()
        self.statusBar.showMessage(f"Solving using {solver}...")
        
        try:
            self.solution = SolverController.Solve(self.level, solver)
            
            if self.solution:
                self.level.grid = self.solution
                self.boardWidget.setLevel(self.level, True)
                self.statusBar.showMessage(f"Solution found using {solver}")
                self.saveButton.setEnabled(True)
            else:
                self.statusBar.showMessage(f"No solution found using {solver}")
                QMessageBox.information(self, "Result", "No solution found.")
                
        except Exception as e:
            self.statusBar.showMessage(f"Error solving puzzle: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error solving puzzle: {str(e)}")
            
    def saveSolution(self):
        """Save the current solution."""
        if not self.solution:
            QMessageBox.warning(self, "Warning", "No solution to save.")
            return
            
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Solution", "source/Outputs", "Text Files (*.txt)"
        )
        
        if fileName:
            try:
                # Update the level number based on the filename
                import re
                match = re.search(r'output-(\d+)', fileName)
                if match:
                    self.level.number = int(match.group(1))
                    
                FileController.WriteSolutionToFile(self.level)
                self.statusBar.showMessage(f"Solution saved to {fileName}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving solution: {str(e)}")

def run():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()