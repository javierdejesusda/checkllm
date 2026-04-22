from __future__ import annotations

import csv
import io
import json

from checkllm.models import CheckResult


class DataCompyMetric:
    """Compare tabular/structured outputs (CSV, JSON arrays).

    Parses both output and reference as tables, then computes column
    match rate, row match rate, and cell-level accuracy. Returns a
    composite score.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        """Initialize the metric.

        Args:
            threshold: Minimum composite score to pass.
        """
        self.threshold = threshold

    def _parse_table(self, text: str) -> tuple[list[str], list[list[str]]]:
        """Parse text as either JSON array or CSV into columns and rows.

        Args:
            text: Input text in CSV or JSON array format.

        Returns:
            Tuple of (column_names, rows) where rows are lists of strings.

        Raises:
            ValueError: If the text cannot be parsed as either format.
        """
        text = text.strip()

        if text.startswith("["):
            try:
                data = json.loads(text)
                if not data or not isinstance(data, list):
                    raise ValueError("JSON is not a non-empty list")

                if isinstance(data[0], dict):
                    columns = list(data[0].keys())
                    rows = [[str(row.get(c, "")) for c in columns] for row in data]
                    return columns, rows
                elif isinstance(data[0], list):
                    columns = [str(c) for c in data[0]]
                    rows = [[str(cell) for cell in row] for row in data[1:]]
                    return columns, rows
                else:
                    raise ValueError("JSON array elements must be dicts or lists")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}") from e

        reader = csv.reader(io.StringIO(text))
        all_rows = list(reader)
        if not all_rows:
            raise ValueError("Empty CSV data")

        columns = all_rows[0]
        rows = all_rows[1:]
        return columns, rows

    async def evaluate(
        self,
        output: str,
        reference: str,
    ) -> CheckResult:
        """Evaluate tabular output against a reference table.

        Computes column match rate, row match rate, and cell-level
        accuracy. The composite score is a weighted average.

        Args:
            output: The model output containing tabular data.
            reference: The reference tabular data.

        Returns:
            CheckResult with composite tabular comparison score.
        """
        try:
            out_cols, out_rows = self._parse_table(output)
        except ValueError as e:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Failed to parse output as table: {e}",
                cost=0.0,
                latency_ms=0,
                metric_name="datacompy_score",
                input_preview=output[:200],
            )

        try:
            ref_cols, ref_rows = self._parse_table(reference)
        except ValueError as e:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Failed to parse reference as table: {e}",
                cost=0.0,
                latency_ms=0,
                metric_name="datacompy_score",
                input_preview=output[:200],
            )

        out_col_set = set(c.lower().strip() for c in out_cols)
        ref_col_set = set(c.lower().strip() for c in ref_cols)
        if ref_col_set:
            col_match = len(out_col_set & ref_col_set) / len(ref_col_set)
        else:
            col_match = 1.0 if not out_col_set else 0.0

        out_row_strs = {tuple(cell.strip() for cell in row) for row in out_rows}
        ref_row_strs = {tuple(cell.strip() for cell in row) for row in ref_rows}
        if ref_row_strs:
            row_match = len(out_row_strs & ref_row_strs) / len(ref_row_strs)
        else:
            row_match = 1.0 if not out_row_strs else 0.0

        total_cells = 0
        matching_cells = 0
        max_rows = min(len(out_rows), len(ref_rows))
        max_cols = min(len(out_cols), len(ref_cols))
        for i in range(max_rows):
            for j in range(max_cols):
                total_cells += 1
                out_val = out_rows[i][j].strip() if j < len(out_rows[i]) else ""
                ref_val = ref_rows[i][j].strip() if j < len(ref_rows[i]) else ""
                if out_val == ref_val:
                    matching_cells += 1

        cell_accuracy = matching_cells / total_cells if total_cells > 0 else 0.0

        composite = 0.3 * col_match + 0.3 * row_match + 0.4 * cell_accuracy
        composite = max(0.0, min(1.0, composite))
        passed = composite >= self.threshold

        return CheckResult(
            passed=passed,
            score=composite,
            reasoning=(
                f"DataCompy: {composite:.4f} "
                f"(col_match: {col_match:.4f}, row_match: {row_match:.4f}, "
                f"cell_accuracy: {cell_accuracy:.4f}, threshold: {self.threshold})"
            ),
            cost=0.0,
            latency_ms=0,
            metric_name="datacompy_score",
            input_preview=output[:200],
        )
