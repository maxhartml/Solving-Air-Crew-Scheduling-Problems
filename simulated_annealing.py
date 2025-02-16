import random
from typing import List, Tuple
import math
from spp_problem import SPPProblem

class SimulatedAnnealing:
    """
    Simulated Annealing solver for the Set Partitioning Problem.
    Uses a penalty-based approach for constraint handling.
    
    Attributes:
    -----------
    problem : SPPProblem
    temp : float
        Initial temperature.
    alpha : float
        Cooling rate (0 < alpha < 1).
    max_iter : int
        Maximum number of iterations (or temperature steps).
    penalty_factor : float
        The penalty factor for each row violation.
    """

    def __init__(self,
                 problem: SPPProblem,
                 temp: float,
                 alpha: float,
                 max_iter: int,
                 penalty_factor: float):
        
        self.problem = problem
        self.temp = temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.penalty_factor = penalty_factor

    def initial_solution(self) -> List[int]:
        """
        Generate a random binary solution of length num_cols.
        Alternatively, you could implement a more clever initialization
        (e.g., partial greedy).
        """
        return [random.randint(0, 1) for _ in range(self.problem.num_cols)]

    def neighbor(self, solution: List[int]) -> List[int]:
        """
        Create a neighbor solution by flipping one or more bits randomly.
        E.g., flip exactly one bit. 
        Could also adopt a more advanced approach.
        """
        new_sol = solution[:]
        j = random.randrange(self.problem.num_cols)
        new_sol[j] = 1 - new_sol[j]  # flip
        return new_sol

    def run(self) -> Tuple[List[int], float]:
        """
        Run Simulated Annealing and return (best_solution, best_fitness).
        """
        current_sol = self.initial_solution()
        current_fitness = self.problem.penalty_function(current_sol, self.penalty_factor)

        best_sol = current_sol
        best_fitness = current_fitness

        T = self.temp

        for iteration in range(self.max_iter):
            # Generate neighbor
            cand_sol = self.neighbor(current_sol)
            cand_fitness = self.problem.penalty_function(cand_sol, self.penalty_factor)

            # Accept if better OR with Metropolis criterion
            if cand_fitness < current_fitness:
                current_sol = cand_sol
                current_fitness = cand_fitness
            else:
                # Accept with probability e^(-(cand_fitness - current_fitness)/T)
                delta = cand_fitness - current_fitness
                accept_prob = math.exp(-delta / T)
                if random.random() < accept_prob:
                    current_sol = cand_sol
                    current_fitness = cand_fitness

            # Update best if improved
            if current_fitness < best_fitness:
                best_sol = current_sol
                best_fitness = current_fitness

            # Cool down
            T *= self.alpha

        return best_sol, best_fitness