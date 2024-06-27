"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver


from typing import Dict 

# 2024.05.15 @hslee
# Dictionary type TASKS (key : str, value : BaseSolver)
TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
}