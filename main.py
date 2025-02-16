#!/usr/bin/env python3
"""
main.py

Demonstrates how to:
1) Run Simulated Annealing (SA), Standard BGA, and Improved BGA on the Set Partitioning Problem.
2) Load each of the three problems (sppnw41.txt, sppnw42.txt, sppnw43.txt).
3) Repeat each algorithm multiple times (e.g. 30) per problem, aggregating metrics.
4) Print results to console AND save them to CSV files.

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
import csv
import os
from typing import Any, Dict, List
from algos.spp_problem import SPPProblem
from algos.simulated_annealing import SimulatedAnnealing
from algos.standard_bga import StandardBGA
from algos.improved_bga import ImprovedBGA


def run_algorithm_multiple_times(
    alg_name: str,
    alg_constructor: Any,
    problem_file: str,
    runs: int = 30
) -> Dict[str, Any]:
    """
    A DRY utility function to:
    1) Load SPP data from problem_file,
    2) Construct the given algorithm multiple times (runs),
    3) Run it, measure time, gather best fitness, feasibility, etc.,
    4) Print aggregated metrics (best/worst/mean/median fitness, feasibility rate, timing),
    5) Save the run-level data + summary stats to a CSV file.

    :param alg_name: Name/label of the algorithm (e.g., "SimulatedAnnealing")
    :param alg_constructor: A callable that, given an SPPProblem instance, returns
                           an algorithm object with a .run() method that yields
                           either (best_sol, best_fit) or (best_sol, best_fit, best_unfit).
    :param problem_file: e.g. "sppnw41.txt"
    :param runs: how many times to run the algorithm
    :return: dictionary of raw data (fitnesses, times, best_solutions, feasibility counts)
    """

    # 1) Load the SPP file
    spp_problem = SPPProblem.from_file(problem_file)

    all_fitnesses: List[float] = []
    all_times: List[float] = []
    feasible_count = 0
    best_solutions: List[List[int]] = []

    # We'll store run-by-run data for CSV
    rows_for_csv = []

    # 2) Repeatedly run the algorithm
    for run_id in range(runs):
        algo = alg_constructor(spp_problem)

        start_time = time.time()
        result = algo.run()
        end_time = time.time()
        elapsed = end_time - start_time

        # handle different return shapes
        if len(result) == 2:
            # (best_sol, best_fit)
            best_sol, best_fit = result
            best_unfit = None
        else:
            # (best_sol, best_fit, best_unfit)
            best_sol, best_fit, best_unfit = result

        feasible, violations = spp_problem.feasibility_and_violations(best_sol)
        if feasible:
            feasible_count += 1

        all_fitnesses.append(best_fit)
        all_times.append(elapsed)
        best_solutions.append(best_sol)

        # Save run-level data for CSV
        rows_for_csv.append([
            run_id + 1,
            best_fit,
            elapsed,
            feasible,
            violations if not feasible else 0,
            best_unfit if best_unfit is not None else "",
        ])

    # 3) Compute summary stats
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
   
    # 5) Write to CSV
    # We'll produce a CSV name like "SimulatedAnnealing_sppnw41.txt_runs30.csv"
    # but you can choose your own naming convention
    base_problem_file = os.path.basename(problem_file)
    csv_filename = f"{alg_name}_{base_problem_file}_runs{runs}.csv"
    path = os.path.join("results", csv_filename)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "Run",
            "BestFitness",
            "TimeSec",
            "Feasible?",
            "Violations",
            "BestUnfit(if any)"
        ])
        # Write each run's row
        for row in rows_for_csv:
            writer.writerow(row)

        # Write summary rows
        writer.writerow([])
        writer.writerow(["Summary Stats", ""])
        writer.writerow(["BestFitness", min_fit])
        writer.writerow(["WorstFitness", max_fit])
        writer.writerow(["MeanFitness", mean_fit])
        writer.writerow(["StDevFitness", stdev_fit])
        writer.writerow(["MedianFitness", median_fit])
        writer.writerow([])
        writer.writerow(["Feasibility(%)", feasibility_rate])
        writer.writerow([])
        writer.writerow(["MinTimeSec", min_time])
        writer.writerow(["MaxTimeSec", max_time])
        writer.writerow(["MeanTimeSec", mean_time])
        writer.writerow(["StDevTimeSec", stdev_time])
        writer.writerow(["MedianTimeSec", median_time])

        print(f'Written to "{csv_filename}"')


def run_sa_on_three_problems(runs: int = 30):
    """
    Runs SimulatedAnnealing multiple times on each of the 3 OR-Library files.
    Then prints & saves CSV with metrics.
    """
    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

    for pf in problem_files:
        def sa_constructor(prob: SPPProblem):
            return SimulatedAnnealing(
                problem=prob,
                temp=1000.0,
                alpha=0.98,
                max_iter=500,  # Adjust as needed
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
    Then prints & saves CSV with metrics.
    """
    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

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
    Then prints & saves CSV with metrics.
    """
    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

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
    # Example: run each algorithm on the 3 problems, 30 runs each.
    run_sa_on_three_problems(runs=30)
    run_standard_bga_on_three_problems(runs=30)
    run_improved_bga_on_three_problems(runs=3)


if __name__ == "__main__":
    main()