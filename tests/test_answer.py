from orchestraos.graph.answer import failure_notice, reconcile_images, strip_outer_fence
from orchestraos.graph.supervisor import format_results

CHART = "outputs/abc/chart.png"


def test_keeps_real_images_and_normalizes_their_path():
    answer = f"Intro\n\n![Sales](./{CHART})\n\nEnd"
    assert reconcile_images(answer, [CHART]) == f"Intro\n\n![Sales]({CHART})\n\nEnd"


def test_drops_made_up_images():
    answer = "Intro\n\n![Trend](path/from/results.png)\n\nEnd"
    assert reconcile_images(answer, []) == "Intro\n\nEnd"


def test_appends_charts_the_writer_forgot():
    result = reconcile_images("Report", [CHART, "outputs/abc/table.csv"])
    assert result == f"Report\n\n## Charts\n\n![chart]({CHART})"


def test_leaves_answers_without_images_alone():
    assert reconcile_images("Just text [1].", []) == "Just text [1]."


def test_failed_step_tells_the_writer_not_to_invent_results():
    text = format_results(
        [{"step_id": 2, "agent": "analytics", "ok": False, "output": "Error: KeyError: 'x'",
          "sources": [], "artifacts": ["outputs/abc/chart.png"]}]
    )  # fmt: skip
    assert "(FAILED)" in text and "KeyError" in text and "do not report" in text
    assert "chart.png" not in text  # files from a failed step are not offered to the writer


def test_no_notice_when_everything_worked():
    assert failure_notice([]) == ""


def test_notice_lists_each_failed_step():
    notice = failure_notice([(2, "analytics", "Error: boom")])
    assert "Some steps failed" in notice and "Step 2 (analytics) failed: Error: boom" in notice


def test_unwraps_answer_fenced_as_markdown():
    assert strip_outer_fence("```markdown\n# Report\n\nText\n```") == "# Report\n\nText"


def test_keeps_answers_that_only_contain_code_blocks():
    answer = "Run this:\n```python\nprint(1)\n```"
    assert strip_outer_fence(answer) == answer


def test_unwraps_markdown_fence_followed_by_text():
    answer = "```markdown\n## Units\n\n| a | b |\n```\n\nSources: [1] sales.csv"
    assert strip_outer_fence(answer) == "## Units\n\n| a | b |\n\nSources: [1] sales.csv"
