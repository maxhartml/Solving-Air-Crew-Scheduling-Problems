"""
Demonstrates how to:
1) Run Simulated Annealing (SA), Standard BGA, and Improved BGA on the Set Partitioning Problem.
2) Load each of the three problems (sppnw41.txt, sppnw42.txt, sppnw43.txt).
3) Repeat each algorithm multiple times (e.g., 30) per problem, aggregating metrics.

Dependencies:
- spp_problem.py
- simulated_annealing.py
- standard_bga.py
- improved_bga.py

Usage:
  python main.py

Adjust parameter values and runs as needed.
"""

import time
import statistics
from typing import Any, Dict, List
from spp_problem import SPPProblem
from simulated_annealing import SimulatedAnnealing
from standard_bga import StandardBGA
from improved_bga import ImprovedBGA


def run_algorithm_multiple_times(
    alg_name: str,
    alg_constructor: Any,
    problem_file: str,
    runs: int = 30
) -> Dict[str, Any]:
    """
    A utility function to:
    1) Load SPP data from problem_file,
    2) Construct the given algorithm multiple times,
    3) Run it, measure time, gather best fitness, feasibility, etc.,
    4) Print aggregated metrics (best/worst/mean/median fitness, feasibility rate, timing).

    :param alg_name: Name/label of the algorithm (e.g., "SimulatedAnnealing")
    :param alg_constructor: A callable that, given an SPPProblem instance, returns
                           an algorithm object with a .run() method that yields
                           either (best_sol, best_fit) or (best_sol, best_fit, best_unfit).
    :param problem_file: e.g. "sppnw41.txt"
    :param runs: how many times to run the algorithm
    :return: dictionary of raw data (fitnesses, times, etc.) for further processing
    """
    # 1) Load the SPP file
    spp_problem = SPPProblem.from_file(problem_file)

    # Collect data
    all_fitnesses = []
    all_times = []
    feasible_count = 0
    best_solutions = []

    # 2) Repeatedly run the algorithm
    for _ in range(runs):
        algo = alg_constructor(spp_problem)

        start_time = time.time()
        result = algo.run()
        end_time = time.time()
        elapsed = end_time - start_time

        # handle different return shapes
        if len(result) == 2:
            best_sol, best_fit = result
        else:
            best_sol, best_fit, best_unfit = result

        feasible, violations = spp_problem.feasibility_and_violations(best_sol)
        if feasible:
            feasible_count += 1

        all_fitnesses.append(best_fit)
        all_times.append(elapsed)
        best_solutions.append(best_sol)

    # 3) Aggregate metrics
    mean_fit = statistics.mean(all_fitnesses)
    stdev_fit = statistics.pstdev(all_fitnesses) if len(all_fitnesses) > 1 else 0
    min_fit = min(all_fitnesses)
    max_fit = max(all_fitnesses)
    median_fit = statistics.median(all_fitnesses)

    mean_time = statistics.mean(all_times)
    stdev_time = statistics.pstdev(all_times) if len(all_times) > 1 else 0
    min_time = min(all_times)
    max_time = max(all_times)
    median_time = statistics.median(all_times)

    feasibility_rate = (feasible_count / runs) * 100.0

    # 4) Print summary
    print(f"\n===== {alg_name} on {problem_file} (runs={runs}) =====")
    print(f"  Best Fitness: {min_fit:.3f}")
    print(f"  Worst Fitness: {max_fit:.3f}")
    print(f"  Mean Fitness: {mean_fit:.3f} (StDev={stdev_fit:.3f})")
    print(f"  Median Fitness: {median_fit:.3f}")
    print(f"  Feasibility Rate: {feasibility_rate:.1f}%  ({feasible_count}/{runs})")
    print(f"  Timing [sec]: min={min_time:.3f}, max={max_time:.3f}, mean={mean_time:.3f}, stdev={stdev_time:.3f}, median={median_time:.3f}")

    return {
        "fitnesses": all_fitnesses,
        "times": all_times,
        "best_solutions": best_solutions,
        "feasible_count": feasible_count
    }


def run_sa_on_three_problems(runs: int = 30):
    """
    Runs SimulatedAnnealing multiple times on each of the 3 OR-Library files.
    Adjust parameters as needed.
    """
    problem_files = ["sppnw41.txt", "sppnw42.txt", "sppnw43.txt"]

    for pf in problem_files:
        def sa_constructor(prob: SPPProblem):
            return SimulatedAnnealing(
                problem=prob,
                temp=1000.0,
                alpha=0.98,
                max_iter=500,
                penalty_factor=2000.0
            )

        run_algorithm_multiple_times(
            alg_name="SimulatedAnnealing",
            alg_constructor=sa_constructor,
            problem_file=pf,
            runs=runs
        )


def run_standard_bga_on_three_problems(runs: int = 30):
    """
    Runs StandardBGA multiple times on each of the 3 OR-Library files.
    Adjust parameters as needed.
    """
    problem_files = ["sppnw41.txt", "sppnw42.txt", "sppnw43.txt"]

    for pf in problem_files:
        def bga_constructor(prob: SPPProblem):
            return StandardBGA(
                problem=prob,
                pop_size=50,
                crossover_rate=0.8,
                mutation_rate=0.01,
                max_generations=200,
                penalty_factor=1000.0,
                tournament_k=3
            )

        run_algorithm_multiple_times(
            alg_name="StandardBGA",
            alg_constructor=bga_constructor,
            problem_file=pf,
            runs=runs
        )


def run_improved_bga_on_three_problems(runs: int = 30):
    """
    Runs ImprovedBGA multiple times on each of the 3 OR-Library files.
    Adjust parameters as needed.
    """
    problem_files = ["sppnw41.txt", "sppnw42.txt", "sppnw43.txt"]

    for pf in problem_files:
        def ibga_constructor(prob: SPPProblem):
            return ImprovedBGA(
                problem=prob,
                pop_size=50,
                max_generations=200,
                crossover_rate=0.8,
                base_mutation_rate=0.01,
                p_stochastic_rank=0.5,
                adaptive_mutation_threshold=0.5,
                adaptive_mutation_count=5,
                seed=42
            )

        run_algorithm_multiple_times(
            alg_name="ImprovedBGA",
            alg_constructor=ibga_constructor,
            problem_file=pf,
            runs=runs
        )


def main():
    # Example calls: each algorithm on each problem, 30 runs
    run_sa_on_three_problems(runs=30)
    run_standard_bga_on_three_problems(runs=30)
    run_improved_bga_on_three_problems(runs=30)


if __name__ == "__main__":
    main()