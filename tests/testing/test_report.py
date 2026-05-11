"""Tests for parity report generation."""

from __future__ import annotations

import json

import jax

from hyesg.testing.comparison import ComparisonResult
from hyesg.testing.report import ParityReport, parity_report
from hyesg.testing.tolerance import (
    TIER_ANALYTICAL,
    TIER_DISTRIBUTIONAL,
    TIER_EXACT,
    TIER_MONTE_CARLO,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# ParityReport dataclass
# ---------------------------------------------------------------------------


class TestParityReport:
    """Tests for the ParityReport dataclass itself."""

    def test_empty_report(self):
        """Empty report with no comparisons defaults to passed."""
        report = ParityReport(name="empty")
        assert report.overall_passed
        assert report.comparisons == []

    def test_to_dict_serialisable(self):
        """to_dict produces JSON-serialisable output."""
        report = ParityReport(
            name="test",
            comparisons=[
                ComparisonResult("t1", True, 0.001, 0.01, {"info": "ok"}),
                ComparisonResult("t2", False, 0.5, 0.01),
            ],
            overall_passed=False,
            summary="1/2 passed",
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_fields(self):
        """to_dict contains expected top-level keys."""
        report = ParityReport(name="check", summary="all good")
        d = report.to_dict()
        assert d["name"] == "check"
        assert d["overall_passed"] is True
        assert d["summary"] == "all good"
        assert d["n_comparisons"] == 0
        assert d["comparisons"] == []

    def test_to_dict_comparison_fields(self):
        """Each comparison in to_dict has all required keys."""
        comp = ComparisonResult("test_a", True, 0.1, 0.5, {"key": "val"})
        report = ParityReport(name="d", comparisons=[comp])
        d = report.to_dict()
        c = d["comparisons"][0]
        assert c["test_name"] == "test_a"
        assert c["passed"] is True
        assert c["metric"] == 0.1
        assert c["threshold"] == 0.5
        assert c["details"] == {"key": "val"}


class TestParityReportMarkdown:
    """Tests for to_markdown rendering."""

    def test_markdown_is_string(self):
        """to_markdown returns a string."""
        report = ParityReport(name="md_test")
        md = report.to_markdown()
        assert isinstance(md, str)

    def test_markdown_contains_title(self):
        """Markdown includes the report name in a heading."""
        report = ParityReport(name="MyReport")
        md = report.to_markdown()
        assert "# Parity Report: MyReport" in md

    def test_markdown_passed_status(self):
        """Passing report shows PASSED."""
        report = ParityReport(name="pass", overall_passed=True)
        md = report.to_markdown()
        assert "PASSED" in md

    def test_markdown_failed_status(self):
        """Failing report shows FAILED."""
        report = ParityReport(name="fail", overall_passed=False)
        md = report.to_markdown()
        assert "FAILED" in md

    def test_markdown_table_header(self):
        """Markdown contains a results table."""
        comp = ComparisonResult("t1", True, 0.001, 0.01)
        report = ParityReport(name="tbl", comparisons=[comp])
        md = report.to_markdown()
        assert "| Test |" in md
        assert "| t1 |" in md

    def test_markdown_summary_section(self):
        """Summary appears in the markdown."""
        report = ParityReport(name="sum", summary="All tests OK")
        md = report.to_markdown()
        assert "All tests OK" in md

    def test_markdown_count(self):
        """Passed count is shown."""
        comps = [
            ComparisonResult("a", True, 0.0, 0.1),
            ComparisonResult("b", False, 0.5, 0.1),
            ComparisonResult("c", True, 0.0, 0.1),
        ]
        report = ParityReport(name="cnt", comparisons=comps)
        md = report.to_markdown()
        assert "2/3" in md


# ---------------------------------------------------------------------------
# parity_report function
# ---------------------------------------------------------------------------


class TestParityReportFunction:
    """Tests for the parity_report orchestrator."""

    def test_matching_data_passes_monte_carlo(
        self, matching_result, golden_master
    ):
        """Matching data passes MONTE_CARLO tier."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_MONTE_CARLO
        )
        assert report.overall_passed

    def test_shifted_data_fails_monte_carlo(
        self, shifted_result, golden_master
    ):
        """Shifted data fails MONTE_CARLO tier."""
        report = parity_report(
            shifted_result, golden_master, tolerance=TIER_MONTE_CARLO
        )
        assert not report.overall_passed

    def test_matching_data_passes_exact(self, matching_result, golden_master):
        """Identical data passes EXACT tier."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_EXACT
        )
        assert report.overall_passed

    def test_matching_data_passes_analytical(
        self, matching_result, golden_master
    ):
        """Identical data passes ANALYTICAL tier."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_ANALYTICAL
        )
        assert report.overall_passed

    def test_matching_data_passes_distributional(
        self, matching_result, golden_master
    ):
        """Identical data passes DISTRIBUTIONAL tier."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_DISTRIBUTIONAL
        )
        assert report.overall_passed

    def test_shifted_data_fails_exact(self, shifted_result, golden_master):
        """Shifted data fails EXACT tier."""
        report = parity_report(
            shifted_result, golden_master, tolerance=TIER_EXACT
        )
        assert not report.overall_passed

    def test_report_name_from_golden_master(
        self, matching_result, golden_master
    ):
        """Report name includes golden master name."""
        report = parity_report(matching_result, golden_master)
        assert golden_master.name in report.name

    def test_report_name_from_simulation_result(self, matching_result):
        """Report name is generic when comparing two SimulationResults."""
        report = parity_report(matching_result, matching_result)
        assert report.name == "parity_report"

    def test_comparisons_cover_all_shared_fields(
        self, matching_result, golden_master
    ):
        """All shared model/field pairs generate comparisons."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_EXACT
        )
        # Count shared fields: 5 fields total across 3 models
        expected_fields = 0
        for model in matching_result.outputs:
            if model in golden_master.outputs:
                for field_name in matching_result.outputs[model]:
                    if field_name in golden_master.outputs[model]:
                        expected_fields += 1
        assert len(report.comparisons) == expected_fields

    def test_monte_carlo_generates_two_tests_per_field(
        self, matching_result, golden_master
    ):
        """MONTE_CARLO tier runs moments + KS per field."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_MONTE_CARLO
        )
        n_fields = 0
        for model in matching_result.outputs:
            if model in golden_master.outputs:
                for field_name in matching_result.outputs[model]:
                    if field_name in golden_master.outputs[model]:
                        n_fields += 1
        assert len(report.comparisons) == 2 * n_fields

    def test_distributional_generates_two_tests_per_field(
        self, matching_result, golden_master
    ):
        """DISTRIBUTIONAL tier runs KS + quantiles per field."""
        report = parity_report(
            matching_result, golden_master, tolerance=TIER_DISTRIBUTIONAL
        )
        n_fields = 0
        for model in matching_result.outputs:
            if model in golden_master.outputs:
                for field_name in matching_result.outputs[model]:
                    if field_name in golden_master.outputs[model]:
                        n_fields += 1
        assert len(report.comparisons) == 2 * n_fields

    def test_summary_reports_counts(self, matching_result, golden_master):
        """Summary string includes pass/total counts."""
        report = parity_report(matching_result, golden_master)
        assert "/" in report.summary  # e.g. "10/10 tests passed."

    def test_summary_lists_failures(self, shifted_result, golden_master):
        """Summary lists failed test names when there are failures."""
        report = parity_report(
            shifted_result, golden_master, tolerance=TIER_EXACT
        )
        assert "Failed:" in report.summary

    def test_report_to_dict_roundtrip(self, matching_result, golden_master):
        """Report can be serialised to dict and then to JSON."""
        report = parity_report(matching_result, golden_master)
        d = report.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["overall_passed"] == report.overall_passed

    def test_report_to_markdown_has_table(
        self, matching_result, golden_master
    ):
        """Markdown output contains a results table."""
        report = parity_report(matching_result, golden_master)
        md = report.to_markdown()
        assert "| Test |" in md
