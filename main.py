"""
Overview:
---------
This script demonstrates how to:
  1) Load the Set Partitioning Problem (SPP) from OR-Library files (sppnw41.txt, sppnw42.txt, sppnw43.txt).
  2) Run three algorithms (Simulated Annealing, Standard BGA, Improved BGA) multiple times (default = 30 runs).
  3) Collect and print metrics (best/worst/mean/median fitness, feasibility, timing).
  4) Save detailed per-run data and summary statistics to a CSV file for each (algorithm, problem) pair.

Workflow:
---------
- For each algorithm on each problem file:
  1) The problem data is loaded (num_rows, num_cols, cost array, coverage).
  2) The algorithm is constructed with chosen hyperparameters.
  3) The algorithm is run multiple times, measuring runtime and evaluating feasibility.
  4) Results are aggregated and displayed, then exported to a CSV in the "results/" folder.

Usage:
------
  python main.py

Adjust parameter values, e.g. population sizes, iteration counts, etc., as needed. 
Uncomment or comment out lines in main() for the algorithms you want to run.

Dependencies:
-------------
- spp_problem.py (SPPProblem class)
- simulated_annealing.py (SimulatedAnnealing)
- standard_bga.py (StandardBGA)
- improved_bga.py (ImprovedBGA)

The "results" directory is where CSV outputs will be placed.
"""

import time
import statistics
import csv
import os
from typing import Any, Dict, List

# Local modules
from algorithms.spp_problem import SPPProblem
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.standard_bga import StandardBGA
from algorithms.improved_bga import ImprovedBGA

def run_algorithm_multiple_times(alg_name: str, alg_constructor: Any, problem_file: str, runs: int = 30) -> Dict[str, Any]:
    """
    Runs a given algorithm multiple times on the specified SPP file,
    collecting run-level data and summary metrics, then writes them to CSV.

    :param alg_name: Identifying name of the algorithm (e.g. "SimulatedAnnealing")
    :param alg_constructor: A callable that takes an SPPProblem and returns an algo instance with a .run() method 
    :param problem_file: The path to the SPP data file
    :param runs: How many times to run the algorithm
    :return: Dictionary of results (fitnesses, times, best_solutions, feasible_count)
    """
    # 1) Load problem data
    spp_problem = SPPProblem.from_file(problem_file)

    all_fitnesses: List[float] = []
    all_times: List[float] = []
    feasible_count = 0
    best_solutions: List[List[int]] = []

    # We'll keep run-level info for CSV
    run_rows = []

    # 2) Run the algorithm 'runs' times
    for run_idx in range(runs):
        algo = alg_constructor(spp_problem)

        start_time = time.time()
        result = algo.run()
        end_time = time.time()
        elapsed_sec = end_time - start_time

        # result can be (best_sol, best_fit) or (best_sol, best_fit, best_unfit)
        if len(result) == 2:
            best_sol, best_fit = result
            best_unfit = None
        else:
            best_sol, best_fit, best_unfit = result

        # Check feasibility
        feasible, violations = spp_problem.feasibility_and_violations(best_sol)
        if feasible:
            feasible_count += 1

        all_fitnesses.append(best_fit)
        all_times.append(elapsed_sec)
        best_solutions.append(best_sol)

        # Save run-level info
        run_rows.append([
            run_idx + 1,
            f"{best_fit:.1f}",
            f"{elapsed_sec:.5f}",
            str(feasible),
            violations if not feasible else 0,
            f"{best_unfit:.4f}" if best_unfit is not None else "N/A"
        ])

    # 3) Summary stats
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

    # 4) Print summary to console
    print(f"\n=== {alg_name} on {problem_file} (runs={runs}) ===")
    print(f"  Best fitness:    {min_fit:.4f}")
    print(f"  Worst fitness:   {max_fit:.4f}")
    print(f"  Mean fitness:    {mean_fit:.4f} (stdev={stdev_fit:.4f})")
    print(f"  Median fitness:  {median_fit:.4f}")
    print(f"  Feasibility:     {feasibility_rate:.1f}%  ({feasible_count}/{runs})")
    print(f"  Timing (s):      min={min_time:.4f}, max={max_time:.4f}, mean={mean_time:.4f}, stdev={stdev_time:.4f}, median={median_time:.4f}")

    # 5) Write data/summary to CSV in "results" folder
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists(f"results/{alg_name}"):
        os.makedirs(f"results/{alg_name}")

    base_problem_file = os.path.basename(problem_file)
    csv_filename = f"{alg_name}_{base_problem_file.strip(".txt")}_runs-{runs}.csv"
    outpath = os.path.join(f"results/{alg_name}", csv_filename)

    with open(outpath, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Section 1: Run-level data
        writer.writerow(["==== RUN-LEVEL RESULTS ===="])
        writer.writerow(["Run", "BestFitness", "TimeSec", "Feasible?", "Violations", "BestUnfitness"])
        for row in run_rows:
            writer.writerow(row)

        writer.writerow([])
        
        # Section 2: Summary
        writer.writerow(["==== SUMMARY METRICS ===="])
        writer.writerow(["BestFitness", f"{min_fit:.4f}"])
        writer.writerow(["WorstFitness", f"{max_fit:.4f}"])
        writer.writerow(["MeanFitness", f"{mean_fit:.4f}"])
        writer.writerow(["StDevFitness", f"{stdev_fit:.4f}"])
        writer.writerow(["MedianFitness", f"{median_fit:.4f}"])
        writer.writerow([])
        writer.writerow(["Feasibility", f"{feasibility_rate:.1f}%"])
        writer.writerow([])
        writer.writerow(["MinTimeSec", f"{min_time:.4f}"])
        writer.writerow(["MaxTimeSec", f"{max_time:.4f}"])
        writer.writerow(["MeanTimeSec", f"{mean_time:.4f}"])
        writer.writerow(["StDevTimeSec", f"{stdev_time:.4f}"])
        writer.writerow(["MedianTimeSec", f"{median_time:.4f}"])
        
        writer.writerow([])
        
        # Section 3: Hyperparameters
        writer.writerow(["==== HYPERPARAMETERS ===="])
        if alg_name == "SimulatedAnnealing":
            writer.writerow(["Temp", algo.temp])
            writer.writerow(["Alpha", algo.alpha])
            writer.writerow(["MaxIter", algo.max_iter])
            writer.writerow(["PenaltyFactor", algo.penalty_factor])
            writer.writerow(["RandomSeed", algo.seed])
        elif alg_name == "StandardBGA":
            writer.writerow(["PopSize", algo.pop_size])
            writer.writerow(["CrossoverRate", algo.crossover_rate])
            writer.writerow(["MutationRate", algo.mutation_rate])
            writer.writerow(["MaxGenerations", algo.max_generations])
            writer.writerow(["PenaltyFactor", algo.penalty_factor])
            writer.writerow(["TournamentK", algo.tournament_k])
            writer.writerow(["RandomSeed", algo.seed])
        elif alg_name == "ImprovedBGA":
            writer.writerow(["PopSize", algo.pop_size])
            writer.writerow(["MaxGenerations", algo.max_generations])
            writer.writerow(["CrossoverRate", algo.crossover_rate])
            writer.writerow(["BaseMutationRate", algo.base_mutation_rate])
            writer.writerow(["PStochasticRank", algo.p_stochastic_rank])
            writer.writerow(["AdaptiveMutationThreshold", algo.adaptive_mutation_threshold])
            writer.writerow(["AdaptiveMutationCount", algo.adaptive_mutation_count])
            writer.writerow(["RandomSeed", algo.seed])  

    print(f'CSV saved to: {outpath}')
    return {
        "fitnesses": all_fitnesses,
        "times": all_times,
        "best_solutions": best_solutions,
        "feasible_count": feasible_count
    }


def run_sa_on_three_problems(runs: int = 30):
    """
    Repeat SimulatedAnnealing for each SPP file, then print and save results.
    """
    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

    for pf in problem_files:
        def sa_constructor(prob: SPPProblem):
            return SimulatedAnnealing(
                problem=prob,
                temp=1000.0,
                alpha=0.98,
                max_iter=50_000,
                penalty_factor=1000.0,
                seed=42
            )

        run_algorithm_multiple_times(
            alg_name="SimulatedAnnealing",
            alg_constructor=sa_constructor,
            problem_file=pf,
            runs=runs
        )


def run_standard_bga_on_three_problems(runs: int = 30):
    """
    Repeat StandardBGA for each SPP file, then print and save results.
    """
    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

    for pf in problem_files:
        def bga_constructor(prob: SPPProblem):
            return StandardBGA(
                problem=prob,
                pop_size=50,
                crossover_rate=0.8,
                mutation_rate=0.02,
                max_generations=200,
                penalty_factor=2000.0,
                tournament_k=2,
                seed=42
            )

        run_algorithm_multiple_times(
            alg_name="StandardBGA",
            alg_constructor=bga_constructor,
            problem_file=pf,
            runs=runs
        )


def run_improved_bga_on_three_problems(runs: int = 30):
    """
    Repeat ImprovedBGA for each SPP file, then print and save results.
    """
    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

    for pf in problem_files:
        def ibga_constructor(prob: SPPProblem):
            return ImprovedBGA(
                problem=prob,
                pop_size=50,
                max_generations=200,
                crossover_rate=0.8,
                base_mutation_rate=0.02,
                p_stochastic_rank=0.45,
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
    """
    Main entry point.
    Uncomment calls below as needed.
    """
    run_sa_on_three_problems(runs=30)
    run_standard_bga_on_three_problems(runs=30)
    run_improved_bga_on_three_problems(runs=30)


if __name__ == "__main__":
    main()