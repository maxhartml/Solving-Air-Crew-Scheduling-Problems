import random
import math
from typing import List, Tuple
from spp_problem import SPPProblem

class SimulatedAnnealing:
    """
    Simulated Annealing for the Set Partitioning Problem (SPP).
    Relies on a cost + penalty approach for row-coverage violations.
    """

    def __init__(
        self,
        problem: SPPProblem,
        temp: float,
        alpha: float,
        max_iter: int,
        penalty_factor: float,
        seed: int
    ):
        """
        :param problem: The SPP problem data/loader
        :param temp: Initial temperature
        :param alpha: Cooling rate (0 < alpha < 1)
        :param max_iter: Maximum number of iterations
        :param penalty_factor: Multiplier for coverage violations
        """
        self.problem = problem
        self.temp = temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.penalty_factor = penalty_factor
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

    def initial_solution(self) -> List[int]:
        """
        Generate a random binary solution (0/1 for each column).
        """
        return [random.randint(0, 1) for _ in range(self.problem.num_cols)]

    def neighbor(self, solution: List[int]) -> List[int]:
        """
        Flip one bit at a random column to produce a neighbor.
        """
        new_sol = solution[:]
        j = random.randrange(self.problem.num_cols)
        new_sol[j] = 1 - new_sol[j]
        return new_sol

    def run(self) -> Tuple[List[int], float]:
        """
        Perform SA and return (best_solution, best_fitness).
        """
        current_sol = self.initial_solution()
        current_fitness = self.problem.penalty_function(current_sol, self.penalty_factor)

        best_sol = current_sol
        best_fitness = current_fitness
        T = self.temp

        for _ in range(self.max_iter):
            # Generate neighbor
            cand_sol = self.neighbor(current_sol)
            cand_fitness = self.problem.penalty_function(cand_sol, self.penalty_factor)

            # Accept if better, else accept with Metropolis probability
            if cand_fitness < current_fitness:
                current_sol, current_fitness = cand_sol, cand_fitness
            else:
                delta = cand_fitness - current_fitness
                accept_prob = math.exp(-delta / T)
                if random.random() < accept_prob:
                    current_sol, current_fitness = cand_sol, cand_fitness

            # Update best
            if current_fitness < best_fitness:
                best_sol = current_sol
                best_fitness = current_fitness

            # Cool down
            T *= self.alpha

        return best_sol, best_fitness