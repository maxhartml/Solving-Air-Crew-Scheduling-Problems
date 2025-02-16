from typing import List, Tuple

class SPPProblem:
    """
    Represents a Set Partitioning Problem (SPP) instance with:
      - num_rows, num_cols
      - costs[j] for each column j
      - coverage[j] = list of rows covered by column j
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        costs: List[float],
        coverage: List[List[int]]
    ):
        """
        :param num_rows: Number of rows (flight legs)
        :param num_cols: Number of columns (feasible crew rotations)
        :param costs: Cost for each column j
        :param coverage: coverage[j] is a list of row indices that column j covers
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.costs = costs
        self.coverage = coverage  # coverage[j] => rows covered by column j

    @classmethod
    def from_file(cls, filename: str) -> 'SPPProblem':
        """
        Load SPP data from a file in OR-Library format.
        First line: num_rows num_cols
        Then each column line: cost k row1 row2 ... rowk (1-based indices)
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
                # Convert 1-based row indices to 0-based
                coverage[col_index] = [r - 1 for r in rows_covered]
                col_index += 1

        return cls(num_rows, num_cols, costs, coverage)

    def compute_cost(self, solution: List[int]) -> float:
        """
        Sum the costs of columns chosen (bit=1).
        """
        total = 0.0
        for j, bit in enumerate(solution):
            if bit == 1:
                total += self.costs[j]
        return total

    def feasibility_and_violations(self, solution: List[int]) -> Tuple[bool, int]:
        """
        Check coverage of each row. 
        Return (is_feasible, total_violations).
        A row is 'violated' if covered 0 or >1 times.
        """
        row_cover_count = [0] * self.num_rows
        for j, bit in enumerate(solution):
            if bit == 1:
                for r in self.coverage[j]:
                    row_cover_count[r] += 1

        violations = 0
        for c in row_cover_count:
            if c != 1:
                violations += abs(c - 1)

        is_feasible = (violations == 0)
        return is_feasible, violations

    def penalty_function(self, solution: List[int], penalty_factor: float) -> float:
        """
        Returns cost + penalty_factor * (coverage_violations).
        """
        cost = self.compute_cost(solution)
        _, violations = self.feasibility_and_violations(solution)
        return cost + penalty_factor * float(violations)