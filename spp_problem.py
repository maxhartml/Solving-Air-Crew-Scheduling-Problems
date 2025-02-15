from typing import List, Tuple

class SPPProblem:
    """
    A class to represent the Set Partitioning Problem (SPP).
    """
    def __init__(self, num_rows: int, num_cols: int,
                 costs: List[float], coverage: List[List[int]]):
        """
        Initialize an SPP problem with the given data.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.costs = costs
        self.coverage = coverage  # coverage[j] is a list of row indices

    @classmethod
    def from_file(cls, filename: str) -> 'SPPProblem':
        """
        Class method to load an SPP problem from a text file in OR-Library format.
        """
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            num_rows, num_cols = map(int, first_line.split())

            costs = []
            coverage = [[] for _ in range(num_cols)]

            col_index = 0
            for line in f:
                if not line.strip():
                    continue
                data = list(map(int, line.split()))
                cost_val = data[0]
                k = data[1]
                rows_covered = data[2:2 + k]

                costs.append(cost_val)
                # Convert row indices from 1-based in data file to 0-based
                coverage[col_index] = [r - 1 for r in rows_covered]
                col_index += 1

        return cls(num_rows, num_cols, costs, coverage)

    def compute_cost(self, solution: List[int]) -> float:
        """
        Compute the total cost of a given binary solution.
        solution[j] in {0,1} indicates if column j is chosen.
        """
        total = 0.0
        for j, bit in enumerate(solution):
            if bit == 1:
                total += self.costs[j]
        return total

    def feasibility_and_violations(self, solution: List[int]) -> Tuple[bool, int]:
        """
        Check if the solution covers each row exactly once.
        Returns: (is_feasible, total_violations)
        
        total_violations can measure how many rows are under-covered or over-covered.
        """
        row_cover_count = [0] * self.num_rows
        for j, bit in enumerate(solution):
            if bit == 1:
                for r in self.coverage[j]:
                    row_cover_count[r] += 1

        # Count how many rows are incorrectly covered
        violations = 0
        for c in row_cover_count:
            if c != 1:  # c == 0 or c >= 2 is a violation
                violations += abs(c - 1)

        is_feasible = (violations == 0)
        return is_feasible, violations

    def penalty_function(self, solution: List[int], penalty_factor: float) -> float:
        """
        A simple penalty-based fitness.
        If violations = number of times row coverage deviates from 1 in total,
        we penalize cost + (penalty_factor * violations).
        """
        cost = self.compute_cost(solution)
        is_feasible, violations = self.feasibility_and_violations(solution)
        fitness = cost + penalty_factor * float(violations)
        return fitness
