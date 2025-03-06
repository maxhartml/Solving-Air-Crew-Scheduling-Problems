"""
Overview:
---------
This script demonstrates how to:
  1) Load the Set Partitioning Problem (SPP) from OR-Library files (sppnw41.txt, sppnw42.txt, sppnw43.txt).
  2) Run three algorithms (Simulated Annealing, Standard BGA, Improved BGA) multiple times (default = 30 runs).
  3) Collect and print metrics (best/worst/mean/median fitness, feasibility, timing).
  4) Save detailed per-run data and summary statistics to a CSV file for each (algorithm, problem) pair.
from typing import List, Tuple

Workflow:
---------
- For each algorithm on each problem file:
  1) The problem data is loaded (num_rows, num_cols, cost array, coverage).
  2) The algorithm is constructed with chosen hyperparameters (potentially distinct per problem).
  3) The algorithm is run multiple times, measuring runtime and evaluating feasibility.
  4) Results are aggregated and displayed, then exported to a CSV in the "results/" folder.

Usage:
------
  python main.py

Dependencies:
-------------
- spp_problem.py (SPPProblem class)
- simulated_annealing.py (SimulatedAnnealing)
- standard_bga.py (StandardBGA)
- improved_bga.py (ImprovedBGA)

The "results" directory is where CSV outputs will be placed.
"""

import time
import datetime
import statistics
import csv
import os
from typing import Any, Dict, List

# Local modules
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
    Runs a given algorithm multiple times on the specified SPP file,
    collecting run-level data and summary metrics, then writes them to CSV.

    :param alg_name: e.g. "SimulatedAnnealing", "StandardBGA", "ImprovedBGA"
    :param alg_constructor: A callable that takes an SPPProblem and returns an algo instance with a .run() method
    :param problem_file: Path to the SPP data file
    :param runs: How many times to run the algorithm
    :return: Dictionary of final aggregated results
    """
    # 1) Load problem data
    spp_problem = SPPProblem.from_file(problem_file)

    # Lists for all runs (feasible or not):
    all_fitnesses: List[float] = []
    all_times: List[float] = []
    best_solutions: List[List[int]] = []
    total_violations: int = 0

    # Lists specifically for feasible runs:
    feasible_fitnesses: List[float] = []
    feasible_times: List[float] = []

    feasible_count = 0

    # We'll keep run-level info for CSV
    run_rows = []

    # 2) Run the algorithm 'runs' times
    for run_idx in range(runs):
        print(f"Running {alg_name} on {problem_file} (run {run_idx + 1}/{runs})...")
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
            feasible_fitnesses.append(best_fit)
            feasible_times.append(elapsed_sec)

        all_fitnesses.append(best_fit)
        all_times.append(elapsed_sec)
        best_solutions.append(best_sol)
        total_violations += violations

        # Build the row for this run
        row = [
            run_idx + 1,
            f"{best_fit:.1f}",
            f"{elapsed_sec:.5f}",
            str(feasible),
            violations if not feasible else 0,
        ]
        # Conditionally append best_unfit only for ImprovedBGA
        if alg_name == "ImprovedBGA":
            row.append(f"{best_unfit:.4f}" if best_unfit is not None else "N/A")

        run_rows.append(row)

    # 3) Summary stats (across ALL runs):
    mean_fit_all = statistics.mean(all_fitnesses)
    stdev_fit_all = statistics.pstdev(all_fitnesses) if len(all_fitnesses) > 1 else 0
    min_fit_all = min(all_fitnesses)
    max_fit_all = max(all_fitnesses)
    median_fit_all = statistics.median(all_fitnesses)

    mean_time_all = statistics.mean(all_times)
    stdev_time_all = statistics.pstdev(all_times) if len(all_times) > 1 else 0
    min_time_all = min(all_times)
    max_time_all = max(all_times)
    median_time_all = statistics.median(all_times)

    feasibility_rate = (feasible_count / runs) * 100.0
    average_violations = total_violations / runs

    # 4) Print summary (all runs) to console
    print(f"\n=== {alg_name} on {problem_file} (runs={runs}) ===")
    print(f"  Feasible runs:   {feasible_count} / {runs} ({feasibility_rate:.1f}%)")
    print(f"  *All runs* best fitness:    {min_fit_all:.4f}")
    print(f"  *All runs* worst fitness:   {max_fit_all:.4f}")
    print(f"  *All runs* mean fitness:    {mean_fit_all:.4f} (stdev={stdev_fit_all:.4f})")
    print(f"  *All runs* median fitness:  {median_fit_all:.4f}")
    print(f"  *All runs* timing (s):      min={min_time_all:.4f}, max={max_time_all:.4f}, mean={mean_time_all:.4f}, stdev={stdev_time_all:.4f}, median={median_time_all:.4f}")
    print(f"  AverageViolations (all runs): {average_violations:.1f}")

    # 5) Summary stats (FEASIBLE-ONLY)
    #    (Compute them only if we have at least one feasible run)
    min_fit_feas, max_fit_feas = None, None
    mean_fit_feas, stdev_fit_feas = None, None
    median_fit_feas = None

    min_time_feas, max_time_feas = None, None
    mean_time_feas, stdev_time_feas = None, None
    median_time_feas = None

    if feasible_fitnesses:
        min_fit_feas = min(feasible_fitnesses)
        max_fit_feas = max(feasible_fitnesses)
        mean_fit_feas = statistics.mean(feasible_fitnesses)
        stdev_fit_feas = statistics.pstdev(feasible_fitnesses) if len(feasible_fitnesses) > 1 else 0
        median_fit_feas = statistics.median(feasible_fitnesses)

        min_time_feas = min(feasible_times)
        max_time_feas = max(feasible_times)
        mean_time_feas = statistics.mean(feasible_times)
        stdev_time_feas = statistics.pstdev(feasible_times) if len(feasible_times) > 1 else 0
        median_time_feas = statistics.median(feasible_times)

        print("\n  **Feasible-only summary**")
        print(f"     Best fitness:       {min_fit_feas:.4f}")
        print(f"     Worst fitness:      {max_fit_feas:.4f}")
        print(f"     Mean fitness:       {mean_fit_feas:.4f} (stdev={stdev_fit_feas:.4f})")
        print(f"     Median fitness:     {median_fit_feas:.4f}")
        print(f"     Timing (s):         min={min_time_feas:.4f}, max={max_time_feas:.4f}, mean={mean_time_feas:.4f}, stdev={stdev_time_feas:.4f}, median={median_time_feas:.4f}")
    else:
        print("\n  **Feasible-only summary**: NO FEASIBLE SOLUTIONS FOUND!")

    # 6) Write data/summary to CSV in "results" folder
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists(f"results/{alg_name}"):
        os.makedirs(f"results/{alg_name}")
   
    base_problem_file = os.path.basename(problem_file)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{alg_name}_{base_problem_file.strip('.txt')}_{timestamp}.csv"
    outpath = os.path.join(f"results/{alg_name}", csv_filename)

    with open(outpath, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Section 1: Run-level data
        writer.writerow(["==== RUN-LEVEL RESULTS ===="])
        # Conditionally write the header
        header = ["Run", "BestFitness", "TimeSec", "Feasible?", "Violations"]
        if alg_name == "ImprovedBGA":
            header.append("BestUnfitness")
        writer.writerow(header)

        # Write the run_rows
        for row in run_rows:
            writer.writerow(row)

        writer.writerow([])

        # Section 2: Summary (ALL RUNS)
        writer.writerow(["==== SUMMARY METRICS (ALL RUNS) ===="])
        writer.writerow(["FeasibleRuns", f"{feasible_count}/{runs} ({feasibility_rate:.1f}%)"])
        writer.writerow(["MinFitness", f"{min_fit_all:.4f}"])
        writer.writerow(["MaxFitness", f"{max_fit_all:.4f}"])
        writer.writerow(["MeanFitness", f"{mean_fit_all:.4f}"])
        writer.writerow(["StDevFitness", f"{stdev_fit_all:.4f}"])
        writer.writerow(["MedianFitness", f"{median_fit_all:.4f}"])
        writer.writerow(["AvgViolations", f"{average_violations:.1f}"])
        writer.writerow([])
        writer.writerow(["MinTimeSec", f"{min_time_all:.4f}"])
        writer.writerow(["MaxTimeSec", f"{max_time_all:.4f}"])
        writer.writerow(["MeanTimeSec", f"{mean_time_all:.4f}"])
        writer.writerow(["StDevTimeSec", f"{stdev_time_all:.4f}"])
        writer.writerow(["MedianTimeSec", f"{median_time_all:.4f}"])

        writer.writerow([])

        # Section 3: Summary (FEASIBLE-ONLY)
        writer.writerow(["==== SUMMARY METRICS (FEASIBLE-ONLY) ===="])
        if feasible_fitnesses:
            writer.writerow(["FeasibleCount", f"{feasible_count}"])
            writer.writerow(["MinFitnessFeasible", f"{min_fit_feas:.4f}"])
            writer.writerow(["MaxFitnessFeasible", f"{max_fit_feas:.4f}"])
            writer.writerow(["MeanFitnessFeasible", f"{mean_fit_feas:.4f}"])
            writer.writerow(["StDevFitnessFeasible", f"{stdev_fit_feas:.4f}"])
            writer.writerow(["MedianFitnessFeasible", f"{median_fit_feas:.4f}"])
            writer.writerow([])
            writer.writerow(["MinTimeSecFeasible", f"{min_time_feas:.4f}"])
            writer.writerow(["MaxTimeSecFeasible", f"{max_time_feas:.4f}"])
            writer.writerow(["MeanTimeSecFeasible", f"{mean_time_feas:.4f}"])
            writer.writerow(["StDevTimeSecFeasible", f"{stdev_time_feas:.4f}"])
            writer.writerow(["MedianTimeSecFeasible", f"{median_time_feas:.4f}"])
        else:
            writer.writerow(["FeasibleSolutionsFound", "0"])

        writer.writerow([])

        # Section 4: Hyperparameters
        writer.writerow(["==== HYPERPARAMETERS ===="])
        # Re-instantiate algo to read hyperparams from it
        example_algo = alg_constructor(spp_problem)
        if alg_name == "SimulatedAnnealing":
            writer.writerow(["Temp", example_algo.temp])
            writer.writerow(["Alpha", example_algo.alpha])
            writer.writerow(["MaxIter", example_algo.max_iter])
            writer.writerow(["PenaltyFactor", example_algo.penalty_factor])
            writer.writerow(["RandomSeed", example_algo.seed])
        elif alg_name == "StandardBGA":
            writer.writerow(["PopSize", example_algo.pop_size])
            writer.writerow(["CrossoverRate", example_algo.crossover_rate])
            writer.writerow(["MutationRate", example_algo.mutation_rate])
            writer.writerow(["MaxGenerations", example_algo.max_generations])
            writer.writerow(["PenaltyFactor", example_algo.penalty_factor])
            writer.writerow(["TournamentK", example_algo.tournament_k])
            writer.writerow(["RandomSeed", example_algo.seed])
        elif alg_name == "ImprovedBGA":
            writer.writerow(["PopSize", example_algo.pop_size])
            writer.writerow(["MaxGenerations", example_algo.max_generations])
            writer.writerow(["CrossoverRate", example_algo.crossover_rate])
            writer.writerow(["BaseMutationRate", example_algo.base_mutation_rate])
            writer.writerow(["PStochasticRank", example_algo.p_stochastic_rank])
            writer.writerow(["AdaptiveMutationThreshold", example_algo.adaptive_mutation_threshold])
            writer.writerow(["AdaptiveMutationCount", example_algo.adaptive_mutation_count])
            writer.writerow(["RandomSeed", example_algo.seed])

    print(f'CSV saved to: {outpath}')

    return {
        "all_fitnesses": all_fitnesses,
        "all_times": all_times,
        "best_solutions": best_solutions,
        "feasible_count": feasible_count,
        "feasible_fitnesses": feasible_fitnesses,
        "feasible_times": feasible_times
    }


def run_sa_on_three_problems(runs: int = 30, problem_files: List[str] = None):
    """
    Repeat SimulatedAnnealing for each SPP file, with distinct hyperparams per file if desired.
    """

    for pf in problem_files:

        # Distinguish hyperparams by file
        if "sppnw41" in pf:
            sa_temp = 2000.0
            sa_alpha = 0.99
            sa_max_iter = 10_000
            sa_penalty_factor = 50_000.0

        elif "sppnw42" in pf:
            sa_temp = 5000.0
            sa_alpha = 0.99
            sa_max_iter = 20_000
            sa_penalty_factor = 100_000.0

        elif "sppnw43" in pf:
            sa_temp = 5000.0
            sa_alpha = 0.99
            sa_max_iter = 20_000
            sa_penalty_factor = 150_000.0

        else:
            print(f"Unknown problem file: {pf}")
            return

        # This constructor will use the adjusted parameters for the given file
        def sa_constructor(prob: SPPProblem):
            return SimulatedAnnealing(
                problem=prob,
                temp=sa_temp,
                alpha=sa_alpha,
                max_iter=sa_max_iter,
                penalty_factor=sa_penalty_factor,
                seed=None
            )

        run_algorithm_multiple_times(
            alg_name="SimulatedAnnealing",
            alg_constructor=sa_constructor,
            problem_file=pf,
            runs=runs
        )


def run_standard_bga_on_three_problems(runs: int = 30, problem_files: List[str] = None):
    """
    Repeat StandardBGA for each SPP file, with distinct hyperparams per file if desired.
    """
    for pf in problem_files:

        # Distinguish hyperparams by file
        if "sppnw41" in pf:
            pop_size = 250
            crossover_rate = 0.8
            mutation_rate = 0.003
            max_generations = 500
            penalty_factor = 4800.0
            tournament_k = 4
        elif "sppnw42" in pf:
            pop_size = 400
            crossover_rate = 0.9
            mutation_rate = 0.003
            max_generations = 1000
            penalty_factor = 10_000.0
            tournament_k = 5
        elif "sppnw43" in pf:
            pop_size = 400
            crossover_rate = 0.9
            mutation_rate = 0.003
            max_generations = 1000
            penalty_factor = 10_000.0
            tournament_k = 5
        else:
            print(f"Unknown problem file: {pf}")
            return

        def bga_constructor(prob: SPPProblem):
            return StandardBGA(
                problem=prob,
                pop_size=pop_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                max_generations=max_generations,
                penalty_factor=penalty_factor,
                tournament_k=tournament_k,
                seed=None
            )

        run_algorithm_multiple_times(
            alg_name="StandardBGA",
            alg_constructor=bga_constructor,
            problem_file=pf,
            runs=runs
        )


def run_improved_bga_on_three_problems(runs: int = 30, problem_files: List[str] = None):
    """
    Repeat ImprovedBGA for each SPP file, with distinct hyperparams if desired.
    """

    for pf in problem_files:

        # Distinguish hyperparams by file
        if "sppnw41" in pf:
            pop_size = 30
            max_gens = 400
            crossover_rate = 0.85
            base_mut_rate = 0.03
            p_stoch_rank = 0.45
            adaptive_threshold = 0.2
            adaptive_count = 2
        elif "sppnw42" in pf:
            pop_size = 30
            max_gens = 400
            crossover_rate = 0.85
            base_mut_rate = 0.03
            p_stoch_rank = 0.45
            adaptive_threshold = 0.25
            adaptive_count = 2
        elif "sppnw43" in pf:
            pop_size = 30
            max_gens = 400
            crossover_rate = 0.85
            base_mut_rate = 0.03
            p_stoch_rank = 0.56
            adaptive_threshold = 0.25
            adaptive_count = 2
        else:
            print(f"Unknown problem file: {pf}")
            return

        def ibga_constructor(prob: SPPProblem):
            return ImprovedBGA(
                problem=prob,
                pop_size=pop_size,
                max_generations=max_gens,
                crossover_rate=crossover_rate,
                base_mutation_rate=base_mut_rate,
                p_stochastic_rank=p_stoch_rank,
                adaptive_mutation_threshold=adaptive_threshold,
                adaptive_mutation_count=adaptive_count,
                seed=None
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
    """

    problem_files = ["data/sppnw41.txt", "data/sppnw42.txt", "data/sppnw43.txt"]

    print(f"Running algorithms on {len(problem_files)} problems...")

    # Run all three approaches with possibly distinct hyperparams per problem
    run_sa_on_three_problems(runs=30, problem_files=problem_files)
    run_standard_bga_on_three_problems(runs=30, problem_files=problem_files)
    run_improved_bga_on_three_problems(runs=30, problem_files=problem_files)


if __name__ == "__main__":
    main()