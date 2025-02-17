from typing import List, Tuple, Optional
import random
from spp_problem import SPPProblem

class StandardBGA:
    """
    A standard binary Genetic Algorithm for SPP using a penalty-based fitness.
    """

    def __init__(
        self,
        problem: SPPProblem,
        pop_size: int,
        crossover_rate: float,
        mutation_rate: float,
        max_generations: int,
        penalty_factor: float,
        tournament_k: int,
        seed: int
    ):
        """
        :param problem: The SPP problem data
        :param pop_size: Population size
        :param crossover_rate: Probability of crossover
        :param mutation_rate: Bit-flip probability
        :param max_generations: Number of evolutionary cycles
        :param penalty_factor: Cost penalty for coverage violations
        :param tournament_k: Size of the k-tournament
        """
        self.problem = problem
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.penalty_factor = penalty_factor
        self.tournament_k = tournament_k
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)


        self.population: List[List[int]] = []
        self.fitnesses: List[float] = []

    def initialize_population(self):
        """
        Create a random population and compute fitness for each individual.
        """
        self.population = []
        for _ in range(self.pop_size):
            individual = [random.randint(0, 1) for _ in range(self.problem.num_cols)]
            self.population.append(individual)

        self.fitnesses = [
            self.problem.penalty_function(ind, self.penalty_factor)
            for ind in self.population
        ]

    def tournament_selection(self) -> List[int]:
        """
        Pick one individual via k-tournament: select k at random, return the best.
        """
        best_ind: Optional[List[int]] = None
        best_fit = float('inf')

        for _ in range(self.tournament_k):
            idx = random.randrange(self.pop_size)
            if self.fitnesses[idx] < best_fit:
                best_fit = self.fitnesses[idx]
                best_ind = self.population[idx]

        return best_ind[:]  # return a copy

    def one_point_crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        One-point crossover: slice both parents at a random point, swap tails.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        point = random.randint(1, self.problem.num_cols - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual: List[int]):
        """
        Bit-flip mutation for each gene with probability=mutation_rate.
        """
        for j in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[j] = 1 - individual[j]

    def run(self) -> Tuple[List[int], float]:
        """
        Execute the GA for max_generations, returning (best_solution, best_fitness).
        """
        self.initialize_population()

        best_sol = None
        best_fit = float('inf')

        for _ in range(self.max_generations):
            new_population = []
            # Generate new offspring until we fill the population
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()

                c1, c2 = self.one_point_crossover(p1, p2)

                self.mutate(c1)
                self.mutate(c2)

                new_population.append(c1)
                new_population.append(c2)

            new_population = new_population[:self.pop_size]

            # Recompute fitness
            self.population = new_population
            self.fitnesses = [
                self.problem.penalty_function(ind, self.penalty_factor)
                for ind in self.population
            ]

            # Track best in this generation
            for i, fit_val in enumerate(self.fitnesses):
                if fit_val < best_fit:
                    best_fit = fit_val
                    best_sol = self.population[i][:]

        return best_sol, best_fit