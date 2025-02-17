# ✈️ Airline Crew Scheduling Project

## 📋 Overview

This project implements three metaheuristics to solve Airline Crew Scheduling problems (modeled as Set Partitioning Problems):
1. **Simulated Annealing (SA)**
2. **Standard Binary Genetic Algorithm (BGA)**
3. **Improved Binary Genetic Algorithm (BGA)**

Each algorithm can be run multiple times on three OR–Library benchmark instances (`sppnw41.txt`, `sppnw42.txt`, `sppnw43.txt`), producing summary metrics and run-level results saved to CSV.

## 📂 File Structure
- **`main.py`**  
    Demonstrates how to run each algorithm (SA, Standard BGA, Improved BGA) on the three benchmark problems.
- **`simulated_annealing.py`, `standard_bga.py`, `improved_bga.py`**  
    Contain the implementations of the three algorithms.
- **`spp_problem.py`**  
    Provides the `SPPProblem` class for loading and managing problem data.
- **`data/`**  
    Folder containing the benchmark instances: `sppnw41.txt`, `sppnw42.txt`, `sppnw43.txt`.
- **`results/`**  
    Created automatically if it doesn’t exist. Stores CSV files containing run-level and summary metrics for each (algorithm, problem) pair.

## 🚀 How to Run
1. **Install Dependencies**
     - Ensure you have Python 3.7+ installed.
     - Install any required and standard libraries (e.g., `time`, `statistics`, `csv`, etc.) are available.
2. **Execute `main.py`**
     - In a terminal or command prompt, navigate to the code folder and run:
         ```sh
         python main.py
         ```
     - By default, the script currently runs only the Simulated Annealing algorithm. You can modify which algorithm(s) to run by uncommenting or commenting the relevant function calls near the bottom of `main.py`:
         ```python
         run_sa_on_three_problems(runs=30)
         # run_standard_bga_on_three_problems(runs=30)
         # run_improved_bga_on_three_problems(runs=30)
         ```
3. **Check Results**
     - CSV outputs will appear in `results/<AlgorithmName>/`, each named according to the algorithm, problem file, and number of runs.
     - Run-level data includes best fitness, time, feasibility, etc. Summary statistics (best, worst, mean, standard deviation) are appended at the end.

## ⚙️ Changing Hyperparameters
- **In `main.py`:**
    - Each algorithm has its own function (`run_sa_on_three_problems`, `run_standard_bga_on_three_problems`, `run_improved_bga_on_three_problems`). Inside these, you’ll see constructors for the algorithms with parameters like `temp`, `alpha`, `pop_size`, `crossover_rate`, etc.
    - Adjust these values to explore different configurations. For example, to change the population size for Standard BGA:
        ```python
        def bga_constructor(prob: SPPProblem):
                return StandardBGA(
                        problem=prob,
                        pop_size=250,          # Adjust here
                        crossover_rate=0.8,
                        ...
                )
        ```
    - You can also modify the number of runs (default = 30) by passing a different `runs` argument to the respective `run_*` function calls.

- **In the Algorithm Classes:**
    - Each algorithm file (`simulated_annealing.py`, `standard_bga.py`, `improved_bga.py`) also contains the logic for how parameters like `penalty_factor`, `mutation_rate`, or `max_iter` are applied. Refer to the docstrings for more details.

## 📝 Additional Notes
- Make sure the `data` folder with `sppnw41.txt`, `sppnw42.txt`, and `sppnw43.txt` is present in the same folder as `main.py`.
- If you want to run everything in one go, simply uncomment all three function calls in `main.py`.
- For best reproducibility, set seed parameters to a fixed integer in the constructors.