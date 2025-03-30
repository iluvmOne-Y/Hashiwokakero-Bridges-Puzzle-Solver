from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, QPoint

import Data.data_Types as _Types

class BoardWidget(QWidget):
    """Widget for displaying the Hashiwokakero puzzle board."""
    
    def __init__(self):
        super().__init__()
        self.level = None
        self.isSolution = False
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def setLevel(self, level: _Types.Level, isSolution: bool = False):
        """Set the level to display.
        
        Args:
            level: The level to display
            isSolution: Whether this is a solution or an initial puzzle
        """
        self.level = level
        self.isSolution = isSolution
        self.update()
        
    def paintEvent(self, event):
        """Paint the board."""
        if not self.level:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get grid size
        rows = self.level.gridSize.y
        cols = self.level.gridSize.x
        
        # Calculate cell size
        w = self.width() / cols
        h = self.height() / rows
        cellSize = min(w, h)
        
        # Center the grid
        offsetX = (self.width() - cellSize * cols) / 2
        offsetY = (self.height() - cellSize * rows) / 2
        
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # Draw grid lines (optional)
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        for i in range(rows + 1):
            painter.drawLine(
                QPointF(offsetX, offsetY + i * cellSize), 
                QPointF(offsetX + cols * cellSize, offsetY + i * cellSize)
            )
        for i in range(cols + 1):
            painter.drawLine(
                QPointF(offsetX + i * cellSize, offsetY),
                QPointF(offsetX + i * cellSize, offsetY + rows * cellSize)
            )
        
        # Draw islands and bridges
        for y in range(rows):
            for x in range(cols):
                cell = self.level.grid[y][x]
                cx = offsetX + x * cellSize + cellSize/2
                cy = offsetY + y * cellSize + cellSize/2
                
                # Draw bridges
                if self.isSolution:
                    if cell == "-":  # Single horizontal bridge
                        painter.setPen(QPen(Qt.black, 3))
                        painter.drawLine(
                            QPointF(cx - cellSize/2, cy), 
                            QPointF(cx + cellSize/2, cy)
                        )
                    elif cell == "=":  # Double horizontal bridge
                        painter.setPen(QPen(Qt.black, 3))
                        painter.drawLine(
                            QPointF(cx - cellSize/2, cy - 3), 
                            QPointF(cx + cellSize/2, cy - 3)
                        )
                        painter.drawLine(
                            QPointF(cx - cellSize/2, cy + 3), 
                            QPointF(cx + cellSize/2, cy + 3)
                        )
                    elif cell == "|":  # Single vertical bridge
                        painter.setPen(QPen(Qt.black, 3))
                        painter.drawLine(
                            QPointF(cx, cy - cellSize/2), 
                            QPointF(cx, cy + cellSize/2)
                        )
                    elif cell == "$":  # Double vertical bridge
                        painter.setPen(QPen(Qt.black, 3))
                        painter.drawLine(
                            QPointF(cx - 3, cy - cellSize/2), 
                            QPointF(cx - 3, cy + cellSize/2)
                        )
                        painter.drawLine(
                            QPointF(cx + 3, cy - cellSize/2), 
                            QPointF(cx + 3, cy + cellSize/2)
                        )
                
                # Draw islands
                if cell not in ["0", "-", "=", "|", "$"]:
                    # Island circle
                    painter.setBrush(QColor(250, 250, 200))
                    painter.setPen(QPen(Qt.black, 2))
                    islandSize = cellSize * 0.7
                    painter.drawEllipse(
                        QRectF(cx - islandSize/2, cy - islandSize/2, islandSize, islandSize)
                    )
                    
                    # Island number
                    painter.setPen(Qt.black)
                    font = QFont()
                    font.setPointSize(int(cellSize/3))
                    font.setBold(True)
                    painter.setFont(font)
                    painter.drawText(
                        QRectF(cx - islandSize/2, cy - islandSize/2, islandSize, islandSize),
                        Qt.AlignCenter,
                        cell
                    )