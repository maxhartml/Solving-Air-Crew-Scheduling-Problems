"""
Implementation of the "Improved BGA" for the Set Partitioning Problem (SPP).
Incorporates:
- Pseudo-random initialization (Algorithm 2 in Chu & Beasley [1])
- Heuristic improvement operator (DROP/ADD, Algorithm 1 in Chu & Beasley [1])
- Stochastic ranking for constraint handling (Runarsson & Yao [2])

References:
[1] P.C. Chu, J.E. Beasley, "Constraint Handling in Genetic Algorithms:
    The Set Partitioning Problem," Journal of Heuristics, 11: 323–357 (1998).
[2] T.P. Runarsson, X. Yao, "Stochastic ranking for constrained evolutionary
    optimization," IEEE Trans. on Evolutionary Computation, 4(3): 284-294 (2000).
"""

import random
from typing import List, Tuple
import math
from algos.spp_problem import SPPProblem

class ImprovedBGA:
    """
    This is a generational GA:
      1) Build initial population using pseudo-random init
      2) Evaluate all (store cost & unfitness separately)
      3) Until max_generations:
         a) Produce offspring using selection, crossover, mutation
         b) Apply the heuristic improvement operator to offspring
         c) Combine parents + offspring
         d) Rank them via "stochastic ranking" (bubble sort approach)
         e) Keep top pop_size solutions
      4) Return the best feasible solution found
    """

    def __init__(self,
                 problem: SPPProblem,
                 pop_size: int = 50,
                 max_generations: int = 200,
                 crossover_rate: float = 0.8,
                 base_mutation_rate: float = 0.01,
                 p_stochastic_rank: float = 0.45,
                 adaptive_mutation_threshold: float = 0.5,
                 adaptive_mutation_count: int = 5,
                 seed: int = None):
        """
        :param problem: The SPP problem instance
        :param pop_size: Population size
        :param max_generations: Number of generations
        :param crossover_rate: Probability of applying crossover
        :param base_mutation_rate: Probability of flipping each bit in standard mutation
        :param p_stochastic_rank: Probability 'p' used in stochastic ranking
        :param adaptive_mutation_threshold: 'epsilon' in [1], if row i is violated in >= epsilon*N of the population,
                                            we forcibly set some bits covering row i to 1
        :param adaptive_mutation_count: 'Ma' in [1], how many bits to flip to 1 for that row
        :param seed: random seed
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

        # We'll store the population as a list of solutions (binary lists).
        # We also keep a separate list for (fitness, unfitness).
        self.population: List[List[int]] = []
        self.fitness_unfitness: List[Tuple[float, float]] = []  # (cost, violation_count)

    # ----------------------------------------------------------------------------
    # 1. Pseudo-Random Initialization (Algorithm 2 in [1], p.341)
    # ----------------------------------------------------------------------------
    def pseudo_random_initialization(self):
        """
        Build each solution in a 'pseudo-random' manner:
         - set solution S = empty
         - let U = set of all rows
         - while U is not empty:
             * pick a row i in U at random
             * pick a column j in alpha(i) that does not over-cover rows not in U
               (i.e. we prefer columns that only cover currently uncovered rows)
             * if j found => add j to the solution => remove from U all rows in beta_j
               (the coverage of j)
             * else => remove i from U (can't cover it now)
        This ensures partial coverage. The solution might still have uncovered rows in
        the end if no feasible column is found, but typically we reduce coverage conflict.
        """
        population = []
        for _ in range(self.pop_size):
            sol = [0]*self.problem.num_cols

            uncovered_rows = set(range(self.problem.num_rows))
            while uncovered_rows:
                i = random.choice(list(uncovered_rows))
                # columns that cover row i
                possible_cols = []
                for j in range(self.problem.num_cols):
                    # If j covers row i
                    if i in self.problem.coverage[j]:
                        # check if picking col j won't cause new coverage for rows outside uncovered_rows
                        # i.e. if coverage[j] is fully within uncovered_rows => no immediate over-coverage
                        # but in practice, [1] allows random approach. We'll do the stricter approach here:
                        covered_rows_j = self.problem.coverage[j]
                        # if all covered rows are in uncovered_rows => no over coverage
                        over_cover = False
                        for r in covered_rows_j:
                            # if r is not in uncovered_rows => picking j covers a row that was covered earlier
                            if r not in uncovered_rows:
                                over_cover = True
                                break
                        if not over_cover:
                            possible_cols.append(j)

                if possible_cols:
                    chosen_j = random.choice(possible_cols)
                    sol[chosen_j] = 1
                    # remove all rows in coverage[j] from uncovered_rows
                    for r in self.problem.coverage[chosen_j]:
                        if r in uncovered_rows:
                            uncovered_rows.remove(r)
                else:
                    # can't cover row i with any column that doesn't cause immediate overlap
                    # so we skip row i
                    uncovered_rows.remove(i)

            population.append(sol)

        self.population = population
        self.fitness_unfitness = [self.evaluate_individual(sol) for sol in population]

    # ----------------------------------------------------------------------------
    # 2. Evaluate (Compute Fitness & Unfitness) as in [1]
    #    "fitness = cost" and "unfitness = sum of coverage deviation"
    # ----------------------------------------------------------------------------
    def evaluate_individual(self, solution: List[int]) -> Tuple[float, float]:
        cost = self.problem.compute_cost(solution)
        feasible, violations = self.problem.feasibility_and_violations(solution)
        return cost, float(violations)  # (fitness, unfitness)

    # ----------------------------------------------------------------------------
    # 3. Stochastic Ranking ([2]) => bubble-sort population by fitness/unfitness
    # ----------------------------------------------------------------------------
    def stochastic_ranking_sort(self, population: List[List[int]],
                                fit_unfit: List[Tuple[float, float]],
                                num_sweeps: int = 2) -> None:
        """
        Perform the "stochastic bubble sort" as per Runarsson & Yao [2].
        For each adjacent pair (i,i+1):
          - If both feasible => compare by cost
          - else => with probability p_stochastic_rank compare by cost,
                    else compare by unfitness
        We do multiple sweeps until stable or hitting num_sweeps.

        :param population: list of solutions
        :param fit_unfit: corresponding list of (fitness, unfitness) for each solution
        :param num_sweeps: how many times we bubble through
        """
        # Typically, you'd do enough sweeps to converge; for demonstration, we do a small number.
        n = len(population)
        for _round in range(num_sweeps):
            swapped = False
            for i in range(n - 1):
                f1, u1 = fit_unfit[i]
                f2, u2 = fit_unfit[i+1]
                # Compare i vs i+1
                # "Both feasible" means both unfitness=0
                if (u1 == 0 and u2 == 0):
                    # Compare by cost => lower cost is "better"
                    if f2 < f1:
                        # swap
                        population[i], population[i+1] = population[i+1], population[i]
                        fit_unfit[i], fit_unfit[i+1] = fit_unfit[i+1], fit_unfit[i]
                        swapped = True
                else:
                    # With probability p => compare by cost
                    # With probability (1-p) => compare by unfitness
                    if random.random() < self.p_stochastic_rank:
                        # compare by cost
                        if f2 < f1:
                            population[i], population[i+1] = population[i+1], population[i]
                            fit_unfit[i], fit_unfit[i+1] = fit_unfit[i+1], fit_unfit[i]
                            swapped = True
                    else:
                        # compare by unfitness => lower unfitness is "better"
                        if u2 < u1:
                            population[i], population[i+1] = population[i+1], population[i]
                            fit_unfit[i], fit_unfit[i+1] = fit_unfit[i+1], fit_unfit[i]
                            swapped = True
            if not swapped:
                break  # no more changes => sorted

    # ----------------------------------------------------------------------------
    # 4. Heuristic Improvement Operator (DROP/ADD) from [1, Algorithm 1]
    # ----------------------------------------------------------------------------
    def heuristic_improvement(self, solution: List[int]) -> None:
        """
        In-place modification of 'solution' with DROP then ADD phases.
        The objective: fix over-covered rows by removing extra columns,
        then fix under-covered rows by adding columns that do not newly over-cover.

        We rely on coverage counters for each row, and do random removal or add.

        Steps:
         1) DROP
         2) ADD
        """
        # We'll track row coverage counts
        row_cover_count = [0]*self.problem.num_rows
        chosen_cols = [j for j, bit in enumerate(solution) if bit == 1]

        for j in chosen_cols:
            for r in self.problem.coverage[j]:
                row_cover_count[r] += 1

        # --- DROP phase ---
        # We'll iterate columns in random order, removing them if they cause row_cover_count >= 2
        # for any row they cover
        tmp_cols = chosen_cols[:]  # copy
        random.shuffle(tmp_cols)
        for j in tmp_cols:
            # check if any row is over-covered by this column
            any_over = False
            for r in self.problem.coverage[j]:
                if row_cover_count[r] >= 2:
                    any_over = True
                    break
            if any_over:
                # remove j
                solution[j] = 0
                for r in self.problem.coverage[j]:
                    row_cover_count[r] -= 1

        # Now we know no row is covered by >= 2 columns => all row_cover_count <= 1

        # --- ADD phase ---
        # Identify under-covered rows => coverage == 0
        uncovered_rows = [r for r, c in enumerate(row_cover_count) if c == 0]

        # We'll attempt to cover them by picking columns that only cover rows that are uncovered
        # and that has best "cost ratio" c_j / |beta_j|. [1] uses that ratio as a heuristic.
        while uncovered_rows:
            r = random.choice(uncovered_rows)
            uncovered_rows.remove(r)

            # find a column j in alpha(r) that covers only uncovered rows, minimize cost/|beta_j|
            candidate_cols = []
            for j in range(self.problem.num_cols):
                if r in self.problem.coverage[j]:
                    # check if picking this column won't over-cover any row
                    can_pick = True
                    covered = self.problem.coverage[j]
                    for row_covered in covered:
                        if row_cover_count[row_covered] >= 1:
                            # it would over-cover
                            can_pick = False
                            break
                    if can_pick:
                        cost_j = self.problem.costs[j]
                        size_j = len(covered)
                        cost_ratio = cost_j / (size_j if size_j > 0 else 1)
                        candidate_cols.append((j, cost_ratio))

            if candidate_cols:
                # pick the j with min cost_ratio
                candidate_cols.sort(key=lambda x: x[1])
                best_j = candidate_cols[0][0]
                solution[best_j] = 1
                for row_covered in self.problem.coverage[best_j]:
                    row_cover_count[row_covered] += 1
                    # if that row was in uncovered_rows, remove it
                    if row_covered in uncovered_rows:
                        uncovered_rows.remove(row_covered)
            # else if no column found that doesn't cause over coverage, we do nothing for row r

    # ----------------------------------------------------------------------------
    # 5. Genetic Operators: Uniform Crossover + "Adaptive" Mutation from [1]
    # ----------------------------------------------------------------------------

    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Uniform crossover: for each bit, pick from parent1 or parent2 with 50% probability.
        We only apply crossover with probability self.crossover_rate.
        """
        if random.random() > self.crossover_rate:
            # No crossover, just copy
            return parent1[:], parent2[:]

        c1 = []
        c2 = []
        for b1, b2 in zip(parent1, parent2):
            if random.random() < 0.5:
                c1.append(b1)
                c2.append(b2)
            else:
                c1.append(b2)
                c2.append(b1)
        return c1, c2

    def adaptive_mutation(self, child: List[int],
                          population: List[List[int]]) -> None:
        """
        As described in [1], we do:
         - standard bit-flip with some small rate (base_mutation_rate)
         - "adaptive" approach: if >= eps*N individuals in population
           violate row i, forcibly set some columns covering i to 1 in child
        """
        # 1) Standard bit-flip
        for j in range(len(child)):
            if random.random() < self.base_mutation_rate:
                child[j] = 1 - child[j]

        # 2) Gather row violation stats in the current population
        # Count how many solutions violate each row
        # For efficiency, you might precompute it once per iteration
        # but for demonstration, we do it inline.
        N = len(population)
        row_violation_count = [0]*self.problem.num_rows
        for sol in population:
            _, v = self.problem.feasibility_and_violations(sol)
            if v > 0:
                # For each row that is not covered exactly once, we'd need a more detailed approach
                # e.g. to detect if row is under-covered. This is somewhat approximate.
                # Let's do a simpler approach: if rowcovercount(r) !=1 => row is violated.
                row_cover_ct = [0]*self.problem.num_rows
                for jj, bit in enumerate(sol):
                    if bit == 1:
                        for rr in self.problem.coverage[jj]:
                            row_cover_ct[rr] += 1
                for rr, ccount in enumerate(row_cover_ct):
                    if ccount != 1:
                        row_violation_count[rr] += 1

        # Now if row_violation_count[i] >= eps*N => we forcibly set 'Ma' columns that cover row i
        # We'll do that in child.
        for i, count_i in enumerate(row_violation_count):
            if count_i >= self.adaptive_mutation_threshold * N:
                # forcibly set up to self.adaptive_mutation_count columns that cover row i to 1
                # pick columns that cover row i randomly
                candidate_cols = []
                for j in range(self.problem.num_cols):
                    if i in self.problem.coverage[j]:
                        candidate_cols.append(j)
                random.shuffle(candidate_cols)
                # flip up to 'Ma' columns for that row to 1
                for j_col in candidate_cols[:self.adaptive_mutation_count]:
                    child[j_col] = 1

    # ----------------------------------------------------------------------------
    # 6. Selection and Main Loop
    #    We'll do simple binary tournament on the stoch-rank-sorted population.
    # ----------------------------------------------------------------------------

    def run(self) -> Tuple[List[int], float, float]:
        """
        Run the improved BGA for max_generations, returning:
          (best_solution, best_fitness, best_unfitness).
        """
        # Step 1: Pseudo-random initialization
        self.pseudo_random_initialization()

        best_sol = None
        best_fit = float('inf')
        best_unfit = float('inf')

        # generational GA
        for gen in range(self.max_generations):
            # We create an offspring population => pop_size
            offspring_pop = []
            offspring_fit_unfit = []

            # We'll do a simple approach: for each pair in pop_size//2
            # pick parents, do crossover, mutate, improvement
            # but first we rank the population stochastically

            self.stochastic_ranking_sort(self.population,
                                         self.fitness_unfitness,
                                         num_sweeps=2)

            # after sorting, "best" solutions are at the front
            # We'll do a small binary tournament from the front or something simpler:
            for _ in range(self.pop_size // 2):
                p1_idx = random.randint(0, self.pop_size-1)
                p2_idx = random.randint(0, self.pop_size-1)
                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]

                child1, child2 = self.uniform_crossover(parent1, parent2)

                # Apply adaptive mutation
                self.adaptive_mutation(child1, self.population)
                self.adaptive_mutation(child2, self.population)

                # Apply heuristic improvement
                self.heuristic_improvement(child1)
                self.heuristic_improvement(child2)

                # Evaluate
                f1, u1 = self.evaluate_individual(child1)
                f2, u2 = self.evaluate_individual(child2)

                offspring_pop.append(child1)
                offspring_pop.append(child2)
                offspring_fit_unfit.append((f1, u1))
                offspring_fit_unfit.append((f2, u2))

            # Now combine parents + offspring => total 2*pop_size
            combined_pop = self.population + offspring_pop
            combined_fu = self.fitness_unfitness + offspring_fit_unfit

            # Stochastic rank again
            self.stochastic_ranking_sort(combined_pop, combined_fu, num_sweeps=3)

            # Keep top pop_size
            new_pop = combined_pop[:self.pop_size]
            new_fu = combined_fu[:self.pop_size]

            self.population = new_pop
            self.fitness_unfitness = new_fu

            # track best feasible
            for (sol, (fit, unfit)) in zip(self.population, self.fitness_unfitness):
                if unfit == 0 and fit < best_fit:
                    best_fit = fit
                    best_unfit = 0.0
                    best_sol = sol[:]

        return (best_sol, best_fit, best_unfit)
