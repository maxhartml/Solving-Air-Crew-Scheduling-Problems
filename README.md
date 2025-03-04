# ✈️ Airline Crew Scheduling Project

## 📋 Overview

This project implements three metaheuristics to solve Airline Crew Scheduling problems (modelled as Set Partitioning Problems):
1. **Simulated Annealing (SA)**
2. **Standard Binary Genetic Algorithm (BGA)**
3. **Improved Binary Genetic Algorithm (BGA)**

Each algorithm can be run multiple times on three OR–Library benchmark instances (`sppnw41.txt`, `sppnw42.txt`, `sppnw43.txt`), producing summary metrics and run-level results saved to CSV files.

## 📂 File Structure

- **`main.py`**  
  Demonstrates how to run each algorithm (SA, Standard BGA, Improved BGA) on the three benchmark problems.

- **`simulated_annealing.py`, `standard_bga.py`, `improved_bga.py`**  
  Contain the implementations of the three algorithms.

- **`spp_problem.py`**  
  Provides the `SPPProblem` class for loading and managing problem data.

- **`data/`**  
  Directory containing the benchmark instances: `sppnw41.txt`, `sppnw42.txt`, `sppnw43.txt`.

- **`results/`**  
  Created automatically if it doesn’t exist. Stores CSV files containing run-level and summary metrics for each (algorithm, problem) pair.

## 🚀 How to Run Locally

1. **Install Dependencies**
   - Ensure you have **Python 3.7+** installed.
   - No external libraries are required beyond the standard library (e.g.\ `typing`, `random`, `math`, `time`, etc.).

2. **Execute `main.py`**
   - In a terminal or command prompt, navigate to the code folder and run:
     ```bash
     python main.py
     ```
   - By default, the script currently calls only the Simulated Annealing algorithm.  
     Uncomment or comment the relevant function calls near the bottom of `main.py` to run each algorithm as desired:
     ```python
     run_sa_on_three_problems(runs=30)
     # run_standard_bga_on_three_problems(runs=30)
     # run_improved_bga_on_three_problems(runs=30)
     ```

3. **Check Results**
   - CSV outputs will appear in `results/<AlgorithmName>/`, named according to the algorithm, problem file, and timestamp.
   - Each file contains run-level data (best fitness, time, feasibility, etc.) and summary statistics (best, worst, mean, standard deviation).

## 🌐 Running in an Online Python IDE

If you wish to run this code directly in an online Python environment (e.g.\ [repl.it](https://repl.it), [Google Colab](https://colab.research.google.com), or similar):

1. **Upload/Copy All Files**  
   - Place `main.py`, the three algorithm files (`simulated_annealing.py`, `standard_bga.py`, `improved_bga.py`), and `spp_problem.py` into the IDE.  
   - Also upload or create a `data/` directory containing `sppnw41.txt`, `sppnw42.txt`, and `sppnw43.txt`.  
   Make sure the folder structure or file references remain consistent with the imports in `main.py`.

2. **Run `main.py`**  
   - In most online IDEs, you can simply open the `main.py` file and click a “Run” button or type a command like:
     ```bash
     python main.py
     ```
     in a provided console/terminal.

3. **No Additional Libraries Needed**  
   - The code depends only on Python’s **standard library** modules (e.g.\ `random`, `math`, `time`, `csv`, `typing`), which are typically pre-installed in such environments. There is no need to install extra packages with `pip`.

4. **Retrieving Output**  
   - Results are written to CSV files in the `results/` directory. Ensure the online IDE you use supports file download or lets you view generated files. Each CSV is named to reflect the algorithm used and the problem instance.

## ⚙️ Changing Hyperparameters

- **In `main.py`**  
  Each algorithm has its own function (`run_sa_on_three_problems`, `run_standard_bga_on_three_problems`, `run_improved_bga_on_three_problems`). Inside, you’ll see constructors for the algorithms with parameters like `temp`, `alpha`, `pop_size`, `crossover_rate`, etc. You can modify these values to experiment with different configurations.

- **In the Algorithm Classes**  
  Each algorithm file (`simulated_annealing.py`, `standard_bga.py`, `improved_bga.py`) defines how parameters like `penalty_factor`, `mutation_rate`, or `max_iter` are applied. Refer to the docstrings for details.

## 📝 Additional Notes

- **Data Placement**  
  Make sure the `data` folder with `sppnw41.txt`, `sppnw42.txt`, and `sppnw43.txt` is present in the same folder as `main.py`. The code assumes `data/` is a subdirectory of the working directory.

- **Multiple Algorithms**  
  If you want to run all three algorithms automatically, simply uncomment all three calls in `main.py`:
  ```python
  run_sa_on_three_problems(runs=30)
  run_standard_bga_on_three_problems(runs=30)
  run_improved_bga_on_three_problems(runs=30)