"""
An example script that demonstrates usage of:
- Simulated Annealing (SA)
- Standard BGA
- Improved BGA
"""

from spp_problem import SPPProblem
from simulated_annealing import SimulatedAnnealing
from standard_bga import StandardBGA

# You must also import your Improved BGA:
from improved_bga import ImprovedBGA

def example_usage():
    # Suppose we have a file 'sppnw41.txt' in the working directory:
    problem_file = "sppnw43.txt"
    spp_problem = SPPProblem.from_file(problem_file)

    # 1. Run Simulated Annealing
    sa_solver = SimulatedAnnealing(
        problem=spp_problem,
        temp=1000.0,
        alpha=0.98,
        max_iter=150_000,
        penalty_factor=2000.0
    )
    best_sol_sa, best_fit_sa = sa_solver.run()
    print("SA: best fitness =", best_fit_sa)
    feasible_sa, v_sa = spp_problem.feasibility_and_violations(best_sol_sa)
    print("SA: feasible?", feasible_sa, "violations:", v_sa)

    # # 2. Run Standard BGA
    # bga_solver = StandardBGA(
    #     problem=spp_problem,
    #     pop_size=50,
    #     crossover_rate=0.7,
    #     mutation_rate=0.01,
    #     max_generations=200,
    #     penalty_factor=2000.0,
    #     tournament_k=3
    # )
    # best_sol_bga, best_fit_bga = bga_solver.run()
    # print("BGA: best fitness =", best_fit_bga)
    # feasible_bga, v_bga = spp_problem.feasibility_and_violations(best_sol_bga)
    # print("BGA: feasible?", feasible_bga, "violations:", v_bga)

    # # 3. Run Improved BGA
    # ibga_solver = ImprovedBGA(
    #     problem=spp_problem,
    #     pop_size=50,
    #     max_generations=200,
    #     crossover_rate=0.8,
    #     base_mutation_rate=0.01,
    #     p_stochastic_rank=0.5,        # Probability used in stochastic ranking
    #     adaptive_mutation_threshold=0.5,  # If >= 50% of population violate a row, forcibly fix some columns
    #     adaptive_mutation_count=5,        # Force up to 5 columns for that row to 1
    #     seed=42
    # )
    # best_sol_ibga, best_fit_ibga, best_unfit_ibga = ibga_solver.run()
    # print("Improved BGA: best cost =", best_fit_ibga, "unfitness =", best_unfit_ibga)
    # if best_sol_ibga is not None:
    #     feasible_ibga, v_ibga = spp_problem.feasibility_and_violations(best_sol_ibga)
    #     print("Improved BGA: feasible?", feasible_ibga, "violations:", v_ibga)
    # else:
    #     print("Improved BGA: no feasible solution found")

if __name__ == "__main__":
    example_usage()