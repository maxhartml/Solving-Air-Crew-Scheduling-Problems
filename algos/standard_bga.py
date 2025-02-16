from typing import List, Tuple, Optional
import random
from algos.spp_problem import SPPProblem


class StandardBGA:
    """
    A standard binary genetic algorithm for SPP with penalty-based constraint handling.

    Attributes:
    -----------
    problem : SPPProblem
    pop_size : int
    crossover_rate : float
    mutation_rate : float
    max_generations : int
    penalty_factor : float
    tournament_k : int
        Parameter for tournament selection.
    """

    def __init__(self,
                 problem: SPPProblem,
                 pop_size: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.02,
                 max_generations: int = 200,
                 penalty_factor: float = 1000.0,
                 tournament_k: int = 3):
        
        self.problem = problem
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.penalty_factor = penalty_factor
        self.tournament_k = tournament_k

        # Internal placeholders
        self.population: List[List[int]] = []
        self.fitnesses: List[float] = []

    def initialize_population(self):
        """
        Initialize the population randomly.
        """
        self.population = []
        for _ in range(self.pop_size):
            individual = [random.randint(0,1) for _ in range(self.problem.num_cols)]
            self.population.append(individual)

        # Calculate fitnesses
        self.fitnesses = [self.problem.penalty_function(ind, self.penalty_factor)
                          for ind in self.population]

    def tournament_selection(self) -> List[int]:
        """
        Select one individual by a standard k-tournament.
        """
        best: Optional[List[int]] = None
        best_fitness = float('inf')

        for _ in range(self.tournament_k):
            idx = random.randrange(self.pop_size)
            if self.fitnesses[idx] < best_fitness:
                best = self.population[idx]
                best_fitness = self.fitnesses[idx]

        return best[:]

    def one_point_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        One-point crossover on binary lists.
        """
        if random.random() > self.crossover_rate:
            # No crossover
            return parent1[:], parent2[:]
        
        point = random.randint(1, self.problem.num_cols - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual: List[int]):
        """
        Bit-flip mutation for each gene with probability mutation_rate.
        """
        for j in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[j] = 1 - individual[j]

    def run(self) -> Tuple[List[int], float]:
        """
        Run the Genetic Algorithm for max_generations.
        Return the best solution found and its fitness.
        """
        self.initialize_population()

        best_sol = None
        best_fit = float('inf')

        for gen in range(self.max_generations):
            new_population = []
            new_fitnesses = []

            # Elitism or not? For simplicity, none here. But you can add it.
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()

                c1, c2 = self.one_point_crossover(p1, p2)

                self.mutate(c1)
                self.mutate(c2)

                new_population.append(c1)
                new_population.append(c2)

            # If we overfilled slightly
            new_population = new_population[:self.pop_size]

            # Update population
            self.population = new_population
            self.fitnesses = [self.problem.penalty_function(ind, self.penalty_factor)
                              for ind in self.population]

            # Track best
            for i, fit in enumerate(self.fitnesses):
                if fit < best_fit:
                    best_fit = fit
                    best_sol = self.population[i][:]

        return best_sol, best_fit

