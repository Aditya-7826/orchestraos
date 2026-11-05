from orchestraos.graph.answer import (
    drop_unbacked_sources,
    failure_notice,
    reconcile_images,
    strip_outer_fence,
)
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


def test_drops_stray_closing_fence():
    assert strip_outer_fence("# Checklist\n\n- [ ] tests\n```") == "# Checklist\n\n- [ ] tests"


def test_drops_stray_opening_fence():
    assert strip_outer_fence("```markdown\n# Checklist\n- a") == "# Checklist\n- a"


def test_keeps_balanced_code_blocks():
    answer = "Run:\n```bash\nmake up\n```\nDone."
    assert strip_outer_fence(answer) == answer


def test_made_up_source_and_citation_are_removed_when_nothing_was_researched():
    answer = (
        "- LangGraph is a library for agents [1].\n\nSources:\n[1] https://www.langgraph.com/about"
    )
    assert drop_unbacked_sources(answer, []) == "- LangGraph is a library for agents."


def test_real_sources_are_kept_and_fake_ones_dropped():
    answer = (
        "LangGraph [1] vs CrewAI [2].\n\nSources:\n"
        "[1] https://real.example/a\n[2] https://invented.example/b"
    )
    cleaned = drop_unbacked_sources(answer, ["https://real.example/a"])
    assert cleaned == "LangGraph [1] vs CrewAI [2].\n\nSources:\n[1] https://real.example/a"


def test_local_file_sources_dont_count_as_web_citations():
    answer = "Revenue was 106,080 [1].\n\nSources: None"
    assert drop_unbacked_sources(answer, ["sales.csv"]) == "Revenue was 106,080."


def test_markdown_links_to_known_urls_survive():
    answer = "See [the docs](https://real.example/a)."
    assert drop_unbacked_sources(answer, ["https://real.example/a"]) == answer


def test_non_image_files_become_plain_links():
    answer = "Here:\n\n![Filtered Employees](outputs/x/filtered.csv)"
    assert reconcile_images(answer, ["outputs/x/filtered.csv"]) == (
        "Here:\n\n[Filtered Employees](outputs/x/filtered.csv)"
    )
