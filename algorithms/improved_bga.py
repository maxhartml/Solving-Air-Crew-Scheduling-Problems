"""
Improved BGA for the Set Partitioning Problem (SPP).

References:
[1] Chu & Beasley (1998), “Constraint Handling in Genetic Algorithms: The Set Partitioning Problem,” Journal of Heuristics, 11: 323–357.
[2] Runarsson & Yao (2000), “Stochastic ranking for constrained evolutionary optimization,” IEEE Transactions on Evolutionary Computation, 4(3): 284–294.
"""

import random
from typing import List, Tuple
import math
from algorithms.spp_problem import SPPProblem

class ImprovedBGA:
    """
    A generational Genetic Algorithm that integrates:
      - Pseudo-random initialization (Chu & Beasley [1])
      - Heuristic improvement operator [DROP/ADD] (Chu & Beasley [1])
      - Stochastic ranking (Runarsson & Yao [2])
    Returns the best feasible solution found (or None if none found).
    """

    def __init__(
        self,
        problem: SPPProblem,
        pop_size: int,
        max_generations: int,
        crossover_rate: float,
        base_mutation_rate: float,
        p_stochastic_rank: float,
        adaptive_mutation_threshold: float,
        adaptive_mutation_count: int,
        seed: int
    ):
        """
        :param problem: SPPProblem instance with cost and coverage info
        :param pop_size: Population size
        :param max_generations: Number of generations to run
        :param crossover_rate: Probability of performing crossover
        :param base_mutation_rate: Probability of flipping each bit in normal mutation
        :param p_stochastic_rank: Probability (p) used in stochastic ranking sort
        :param adaptive_mutation_threshold: If row is violated in >= threshold*N solutions, forcibly cover it
        :param adaptive_mutation_count: How many columns to set for that row
        :param seed: Random seed for reproducibility
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.base_mutation_rate = base_mutation_rate
        self.p_stochastic_rank = p_stochastic_rank
        self.adaptive_mutation_threshold = adaptive_mutation_threshold
        self.adaptive_mutation_count = adaptive_mutation_count

        if seed is not None:
            random.seed(seed)

        # population: list of binary solutions
        # fitness_unfitness: list of (cost, coverage_violations)
        self.population: List[List[int]] = []
        self.fitness_unfitness: List[Tuple[float, float]] = []

    def pseudo_random_initialization(self):
        """
        Build each solution with partial coverage to avoid immediate overlaps.
        Rows are processed randomly, choosing columns that don't overlap covered rows.
        """
        population = []
        for _ in range(self.pop_size):
            sol = [0] * self.problem.num_cols
            uncovered_rows = set(range(self.problem.num_rows))

            while uncovered_rows:
                row_i = random.choice(list(uncovered_rows))
                possible_cols = []
                # Find columns that cover row_i but do not re-cover any covered row
                for col_j in range(self.problem.num_cols):
                    if row_i in self.problem.coverage[col_j]:
                        covered_rows_j = self.problem.coverage[col_j]
                        over_cover = False
                        # Check if picking this column would cover already-covered rows
                        for r in covered_rows_j:
                            if r not in uncovered_rows:
                                over_cover = True
                                break
                        if not over_cover:
                            possible_cols.append(col_j)

                if possible_cols:
                    chosen_col = random.choice(possible_cols)
                    sol[chosen_col] = 1
                    # Remove rows covered by this column from uncovered_rows
                    for r in self.problem.coverage[chosen_col]:
                        if r in uncovered_rows:
                            uncovered_rows.remove(r)
                else:
                    # Skip row_i if no column can be picked without overlap
                    uncovered_rows.remove(row_i)

            population.append(sol)

        self.population = population
        self.fitness_unfitness = [
            self.evaluate_individual(sol) for sol in population
        ]

    def evaluate_individual(self, solution: List[int]) -> Tuple[float, float]:
        """
        Returns (cost, coverage_violations).
        coverage_violations is how many rows are incorrectly covered (0 or 2+ times).
        """
        cost = self.problem.compute_cost(solution)
        _, violations = self.problem.feasibility_and_violations(solution)
        return cost, float(violations)

    def stochastic_ranking_sort(self, population: List[List[int]], fit_unfit: List[Tuple[float, float]], num_sweeps: int = 2):
        """
        Bubble-sort approach that, for each adjacent pair:
          - If both feasible, compare by cost.
          - Else, with probability p_stochastic_rank compare by cost, else by unfitness.
        This balances cost vs. feasibility when ordering solutions.
        """
        n = len(population)
        for _ in range(num_sweeps):
            swapped = False
            for i in range(n - 1):
                cost1, unfit1 = fit_unfit[i]
                cost2, unfit2 = fit_unfit[i+1]

                if unfit1 == 0 and unfit2 == 0:
                    # Both feasible -> compare cost
                    if cost2 < cost1:
                        population[i], population[i+1] = population[i+1], population[i]
                        fit_unfit[i], fit_unfit[i+1] = fit_unfit[i+1], fit_unfit[i]
                        swapped = True
                else:
                    # With probability p compare by cost, else by unfitness
                    if random.random() < self.p_stochastic_rank:
                        if cost2 < cost1:
                            population[i], population[i+1] = population[i+1], population[i]
                            fit_unfit[i], fit_unfit[i+1] = fit_unfit[i+1], fit_unfit[i]
                            swapped = True
                    else:
                        if unfit2 < unfit1:
                            population[i], population[i+1] = population[i+1], population[i]
                            fit_unfit[i], fit_unfit[i+1] = fit_unfit[i+1], fit_unfit[i]
                            swapped = True
            if not swapped:
                break  # No more swaps => sorted enough

    def heuristic_improvement(self, solution: List[int]):
        """
        DROP/ADD steps:
          1) DROP columns causing over-coverage
          2) ADD columns to cover any uncovered rows (lowest cost ratio).
        """
        row_cover_count = [0] * self.problem.num_rows
        chosen_cols = [j for j, bit in enumerate(solution) if bit == 1]

        for col_j in chosen_cols:
            for r in self.problem.coverage[col_j]:
                row_cover_count[r] += 1

        # DROP: remove columns if row_cover_count[r] >=2 for any r they cover
        shuffled_chosen_cols = chosen_cols[:]
        random.shuffle(shuffled_chosen_cols)
        for col_j in shuffled_chosen_cols:
            any_over = False
            for r in self.problem.coverage[col_j]:
                if row_cover_count[r] >= 2:
                    any_over = True
                    break
            if any_over:
                solution[col_j] = 0
                for r in self.problem.coverage[col_j]:
                    row_cover_count[r] -= 1

        # ADD: if row_cover_count[r] == 0 => pick column with best cost ratio
        uncovered_rows = [r for r, c in enumerate(row_cover_count) if c == 0]
        while uncovered_rows:
            row_i = random.choice(uncovered_rows)
            uncovered_rows.remove(row_i)

            candidate_cols = []
            for col_j in range(self.problem.num_cols):
                if row_i in self.problem.coverage[col_j]:
                    can_pick = True
                    for rr in self.problem.coverage[col_j]:
                        if row_cover_count[rr] >= 1:
                            can_pick = False
                            break
                    if can_pick:
                        cost_j = self.problem.costs[col_j]
                        size_j = len(self.problem.coverage[col_j])
                        cost_ratio = cost_j / (size_j if size_j > 0 else 1)
                        candidate_cols.append((col_j, cost_ratio))

            if candidate_cols:
                candidate_cols.sort(key=lambda x: x[1])
                best_col = candidate_cols[0][0]
                solution[best_col] = 1
                for r_cov in self.problem.coverage[best_col]:
                    row_cover_count[r_cov] += 1
                    if r_cov in uncovered_rows:
                        uncovered_rows.remove(r_cov)

    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Uniform crossover: for each bit, pick from parent1 or parent2 with 50% chance.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        child1, child2 = [], []
        for b1, b2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(b1)
                child2.append(b2)
            else:
                child1.append(b2)
                child2.append(b1)
        return child1, child2

    def adaptive_mutation(self, child: List[int], population: List[List[int]]):
        """
        Bit-flip mutation plus forcing coverage for rows violated by many solutions.
        """
        # 1) Standard bit-flip
        for j in range(len(child)):
            if random.random() < self.base_mutation_rate:
                child[j] = 1 - child[j]

        # 2) Count how many solutions violate each row
        N = len(population)
        row_violation_count = [0] * self.problem.num_rows
        for sol in population:
            _, v = self.problem.feasibility_and_violations(sol)
            if v > 0:
                row_cover_ct = [0]*self.problem.num_rows
                for col_j, bit in enumerate(sol):
                    if bit == 1:
                        for r in self.problem.coverage[col_j]:
                            row_cover_ct[r] += 1
                for r, ccount in enumerate(row_cover_ct):
                    if ccount != 1:
                        row_violation_count[r] += 1

        # If row i is violated in >= threshold*N solutions => forcibly set columns
        for r, count_i in enumerate(row_violation_count):
            if count_i >= self.adaptive_mutation_threshold * N:
                candidate_cols = []
                for col_j in range(self.problem.num_cols):
                    if r in self.problem.coverage[col_j]:
                        candidate_cols.append(col_j)
                random.shuffle(candidate_cols)
                for col_j in candidate_cols[:self.adaptive_mutation_count]:
                    child[col_j] = 1

    def run(self) -> Tuple[List[int], float, float]:
        """
        Main loop: 
         1) Initialize (pseudo-random).
         2) Repeatedly rank population, create offspring, mutate/improve, combine.
         3) Keep best feasible solution found.
        Returns (best_solution, best_cost, best_unfitness).
        """
        self.pseudo_random_initialization()

        best_solution = None
        best_cost = float('inf')
        best_unfitness = float('inf')

        for _ in range(self.max_generations):
            offspring_population = []
            offspring_fitness_unfitness = []

            # Rank existing population stochastically
            self.stochastic_ranking_sort(self.population, self.fitness_unfitness, num_sweeps=2)

            # Produce offspring in pairs
            for _ in range(self.pop_size // 2):
                p1_idx = random.randint(0, self.pop_size - 1)
                p2_idx = random.randint(0, self.pop_size - 1)
                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]

                # Uniform crossover
                child1, child2 = self.uniform_crossover(parent1, parent2)

                # Adaptive mutation
                self.adaptive_mutation(child1, self.population)
                self.adaptive_mutation(child2, self.population)

                # Heuristic improvement
                self.heuristic_improvement(child1)
                self.heuristic_improvement(child2)

                # Evaluate each new child
                c1_cost, c1_unfit = self.evaluate_individual(child1)
                c2_cost, c2_unfit = self.evaluate_individual(child2)

                offspring_population.append(child1)
                offspring_population.append(child2)
                offspring_fitness_unfitness.append((c1_cost, c1_unfit))
                offspring_fitness_unfitness.append((c2_cost, c2_unfit))

            # Combine parents + offspring => 2x pop_size
            combined_population = self.population + offspring_population
            combined_fit_unfit = self.fitness_unfitness + offspring_fitness_unfitness

            # Stochastic rank again, then keep top pop_size
            self.stochastic_ranking_sort(combined_population, combined_fit_unfit, num_sweeps=3)
            new_population = combined_population[:self.pop_size]
            new_fit_unfit = combined_fit_unfit[:self.pop_size]

            self.population = new_population
            self.fitness_unfitness = new_fit_unfit

            # Track best feasible solution (unfitness=0 => feasible)
            for sol, (cost, unfit) in zip(self.population, self.fitness_unfitness):
                if unfit == 0 and cost < best_cost:
                    best_cost = cost
                    best_unfitness = 0.0
                    best_solution = sol[:]

        return best_solution, best_cost, best_unfitness