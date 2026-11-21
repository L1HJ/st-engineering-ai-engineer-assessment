from agent_harness.chat import render_markdown


def test_render_markdown_strips_markers_without_color() -> None:
    rendered = render_markdown(
        "## Status\n**degraded** and `checkout`\n- timeout rate high",
        color=False,
    )

    assert "**" not in rendered
    assert "`" not in rendered
    assert "##" not in rendered
    assert "Status" in rendered
    assert "degraded and checkout" in rendered
    assert "• timeout rate high" in rendered


def test_render_markdown_removes_ansi_sequences_from_model_output() -> None:
    rendered = render_markdown("\x1b[31mred\x1b[0m text", color=False)

    assert "\x1b" not in rendered
    assert rendered == "red text"


def test_render_markdown_keeps_code_block_content_literal() -> None:
    rendered = render_markdown("```\n**not bold**\n```", color=False)

    assert "**not bold**" in rendered


def test_render_markdown_applies_ansi_when_color_enabled() -> None:
    rendered = render_markdown("**degraded** and `checkout`", color=True)

    assert "\x1b[1mdegraded\x1b[0m" in rendered
    assert "\x1b[36mcheckout\x1b[0m" in rendered
