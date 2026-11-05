"""Deterministic clean-up of the final answer, after the writing agent.

Models sometimes embed images that don't exist (made-up or placeholder paths)
or forget charts that were made. Code, not the model, has the last word here:
only files that a step really produced can appear, and every chart produced does.
"""

import re

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".gif")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)>\s]+)>?\s*\)")
_URL = re.compile(r"https?://[^\s)>\]]+")
_CITATION = re.compile(r"\s?\[\d+\](?!\()")  # [1] but not a markdown link [1](...)
_SOURCES_HEADING = re.compile(r"^\s*(?:#+\s*)?\**sources\**:?\**\s*(?:none)?\s*$", re.I)


_SAVE_VERB = re.compile(r"\b(save|saved|saving|write|written|store|stored)\b", re.I)


def drop_save_instructions(text: str, paths: list[str]) -> str:
    """Remove lines where the writer talks about saving to the target file.

    Told not to, the 7B writer still ended a checklist with "Save the above content to
    `notes/demo-checklist.md`." That line then ended up inside the saved file itself. Saving
    is the action agent's job, and it reports what it did separately.
    """
    names = {p for p in paths if p} | {p.rsplit("/", 1)[-1] for p in paths if p}
    if not names:
        return text
    kept = [
        line
        for line in text.splitlines()
        if not (_SAVE_VERB.search(line) and any(name in line for name in names))
    ]
    cleaned = "\n".join(kept).strip()
    # Never empty the text: if every line mentioned the file, it *is* the content.
    return cleaned or text


def drop_unbacked_sources(answer: str, sources: list[str]) -> str:
    """Remove source URLs no step produced, and all citations if nothing was researched.

    With no research step, a small model will still write "[1]" and invent a plausible
    URL. Only URLs that a tool really returned may appear in the answer.
    """
    known = set(sources)
    kept = []
    for line in answer.splitlines():
        urls = _URL.findall(line)
        if urls and not any(url.rstrip(".,;") in known for url in urls):
            continue  # a line citing only made-up URLs
        kept.append(line)
    text = "\n".join(kept)
    if not any(s.startswith("http") for s in sources):
        text = _CITATION.sub("", text)
    # A "Sources:" heading with nothing left under it.
    lines = text.rstrip().splitlines()
    while lines and (_SOURCES_HEADING.match(lines[-1]) or not lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


_OUTER_FENCE = re.compile(r"^```\s*\n(.*)\n```\s*$", re.DOTALL)
_MARKDOWN_FENCE = re.compile(r"```(?:markdown|md)[ \t]*\n(.*?)\n```", re.DOTALL)


def strip_outer_fence(answer: str) -> str:
    """Unwrap markdown the model put inside a code block, so it renders instead of showing
    as code: any ```markdown block, or a plain ``` block that wraps the whole answer."""
    answer = _MARKDOWN_FENCE.sub(lambda m: m.group(1), answer)
    match = _OUTER_FENCE.match(answer.strip())
    answer = match.group(1).strip() if match else answer.strip()
    return drop_unmatched_fence(answer)


def drop_unmatched_fence(answer: str) -> str:
    """Remove a stray opening or closing ``` line (an odd number of fences would
    otherwise turn the rest of the answer into a code block)."""
    lines = answer.splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fences) % 2 == 0:
        return answer
    if lines[fences[-1]].strip() == "```" and fences[-1] == len(lines) - 1:
        del lines[fences[-1]]
    elif lines[fences[0]].strip() in ("```", "```markdown", "```md") and fences[0] == 0:
        del lines[fences[0]]
    return "\n".join(lines).strip()


def reconcile_images(answer: str, artifacts: list[str]) -> str:
    real = set(artifacts)
    embedded: set[str] = set()

    def keep_or_drop(match: re.Match) -> str:
        path = match.group(2).removeprefix("./").removeprefix("workspace/")
        if path not in real:
            return ""  # drop links to files no step produced
        if not path.lower().endswith(IMAGE_SUFFIXES):
            # A real file that isn't an image (e.g. a CSV the analysis saved): a plain link.
            return f"[{match.group(1) or path.rsplit('/', 1)[-1]}]({path})"
        embedded.add(path)
        return f"![{match.group(1)}]({path})"

    cleaned = _IMAGE.sub(keep_or_drop, answer)
    missing = [a for a in artifacts if a.lower().endswith(IMAGE_SUFFIXES) and a not in embedded]
    if missing:
        cleaned = (
            cleaned.rstrip() + "\n\n## Charts\n\n" + "\n\n".join(f"![chart]({p})" for p in missing)
        )
    # Dropping an image can leave a run of blank lines behind.
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def failure_notice(failed: list[tuple[int, str, str]]) -> str:
    """A notice for steps that failed, as (step id, agent, error) tuples; '' if none.

    The writer is told not to report results for failed steps, but small models
    sometimes do anyway. This notice comes from code, so the user always knows.
    """
    if not failed:
        return ""
    lines = [f"> - Step {step} ({agent}) failed: {error}" for step, agent, error in failed]
    return (
        "> **Some steps failed.** Anything below that depends on them is not backed by "
        "real results.\n" + "\n".join(lines)
    )
