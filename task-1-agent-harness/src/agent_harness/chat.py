"""Minimal interactive REPL over the conversation layer.

Run with ``python -m agent_harness.chat``. This is a thin terminal client for
P7a; streaming and cross-process session management remain future work.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from agent_harness.agent import AgentRunner
from agent_harness.config import load_settings
from agent_harness.conversation import ConversationService
from agent_harness.llm import KimiAdapter
from agent_harness.schemas import RunView
from agent_harness.store import (
    ConversationBusyError,
    ConversationNotFoundError,
    ConversationTurnLimitError,
    Database,
    RunEventRecord,
    RunRepository,
)
from agent_harness.tools import ToolRegistry

try:
    import readline
except ImportError:  # pragma: no cover - platform dependent
    readline = None

STATE_FILE = Path(".agent_harness_chat.json")
HISTORY_FILE = Path(".agent_harness_chat_history")
_READLINE_IS_LIBEDIT = readline is not None and "libedit" in (readline.__doc__ or "")
T = TypeVar("T")
HELP_TEXT = """Commands:
  /help                 Show this help.
  /new [title]          Start a new conversation.
  /id                   Print the current conversation ID.
  /history [limit]      Show recent turns (default 10).
  /trace                Show events for the last turn.
  /exit, /quit          Leave the REPL.

Any other input is sent as the next user message. Approval prompts appear inline."""

_USE_COLOR = sys.stdout.isatty()


def _style(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize_terminal_text(text: str) -> str:
    return _CONTROL_RE.sub("", _ESCAPE_RE.sub("", text))


def _inline_markdown(text: str, color: bool) -> str:
    if not color:
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    pieces = []
    for part in re.split(r"(`[^`]*`)", text):
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            pieces.append(f"\x1b[36m{part[1:-1]}\x1b[0m")
            continue
        part = re.sub(r"\*\*(.+?)\*\*", "\x1b[1m\\1\x1b[0m", part)
        part = re.sub(r"__(.+?)__", "\x1b[1m\\1\x1b[0m", part)
        part = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", part)
        pieces.append(part)
    return "".join(pieces)


def render_markdown(text: str, color: bool | None = None) -> str:
    """Render a small, safe Markdown subset for terminal output."""
    use_color = _USE_COLOR if color is None else color

    def paint(code: str, value: str) -> str:
        return f"\x1b[{code}m{value}\x1b[0m" if use_color else value

    output: list[str] = []
    in_code = False
    for raw in _sanitize_terminal_text(text).splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            output.append(paint("2", "  " + raw))
            continue
        if not stripped:
            output.append("")
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            output.append(paint("2", "─" * 40))
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            output.append(paint("1", _inline_markdown(heading.group(2), use_color)))
            continue
        bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", raw)
        if bullet:
            output.append(
                f"{bullet.group(1)}• {_inline_markdown(bullet.group(2), use_color)}"
            )
            continue
        quote = re.match(r"^\s*>\s?(.*)$", raw)
        if quote:
            content = _inline_markdown(quote.group(1), use_color)
            output.append(paint("2", "│ " + content))
            continue
        output.append(_inline_markdown(raw, use_color))
    return "\n".join(output)


def _read_key(fd: int) -> str | None:
    import select

    try:
        first = os.read(fd, 1)
    except OSError:
        return None
    if not first:
        return None
    if first == b"\x1b":
        ready, _, _ = select.select([fd], [], [], 0.03)
        if not ready:
            return "escape"
        rest = os.read(fd, 2)
        if rest == b"[A":
            return "up"
        if rest == b"[B":
            return "down"
        return "escape"
    if first in (b"\r", b"\n"):
        return "enter"
    if first == b"\x03":
        return "cancel"
    try:
        return first.decode("utf-8").lower()
    except UnicodeDecodeError:
        return ""


class Console:
    def step(self, text: str, *, error: bool = False) -> None:
        print(_style(f"  · {text}", "31" if error else "2"))

    def assistant(self, text: str) -> None:
        print(_style("assistant › ", "1;32") + render_markdown(text))

    def status(self, text: str) -> None:
        print(_style(f"── {text} ──────────────────────────", "2"))

    def user_prompt(self) -> str:
        if (
            not _USE_COLOR
            or readline is None
            or _READLINE_IS_LIBEDIT
            or not sys.stdin.isatty()
        ):
            return "you › "
        # \001 and \002 tell readline that the color escapes take no width.
        return "\001\x1b[1;36m\x02you › \001\x1b[0m\x02"

    def event(self, event: RunEventRecord) -> None:
        try:
            payload = json.loads(event.payload_json)
        except json.JSONDecodeError:
            payload = {}
        event_type = event.event_type
        if event_type == "tool_call":
            self.step(
                f"calling {payload.get('tool_name', '?')} "
                f"{_compact_json(payload.get('arguments', {}))}"
            )
        elif event_type == "tool_result":
            self.step(
                f"finished {payload.get('tool_name', '?')}"
                f"{self._result_suffix(payload)}"
            )
        elif event_type == "tool_error":
            error = (payload.get("error") or {}).get("code", "UNKNOWN")
            self.step(
                f"failed {payload.get('tool_name', '?')}: {error}",
                error=True,
            )
        elif event_type == "retry_scheduled":
            attempt = payload.get("attempt")
            suffix = f" (attempt {attempt})" if attempt else ""
            self.step(f"retrying call{suffix}")
        elif event_type == "llm_error":
            self.step(
                f"model call failed: {payload.get('error_code', 'UNKNOWN')}",
                error=True,
            )
        elif event_type == "conversation_compacted":
            self.step("conversation history compacted")
        elif event_type == "conversation_compaction_failed":
            self.step("conversation compaction failed", error=True)
        elif event_type == "context_compacted":
            self.step("turn context compacted")
        elif event_type == "run_state_changed":
            if payload.get("status") == "WAITING_APPROVAL":
                self.step("waiting for approval")

    @staticmethod
    def _result_suffix(payload: dict[str, Any]) -> str:
        observation = payload.get("observation") or {}
        data = observation.get("data") if isinstance(observation, dict) else None
        if not isinstance(data, dict):
            return ""
        if "incident_id" in data:
            return (
                f" → incident {data['incident_id']} created"
                f" ({data.get('severity', '?')})"
            )
        if "service_name" in data and "status" in data:
            return f" → {data['service_name']}: {data['status']}"
        if "results" in data and isinstance(data["results"], list):
            return f" → {len(data['results'])} knowledge result(s)"
        return ""

    def select(
        self,
        question: str,
        options: list[str],
        shortcuts: dict[str, int],
        default: int,
    ) -> int:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return self._select_text(question, options, shortcuts, default)
        import termios
        import tty

        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        selected = 0
        try:
            tty.setcbreak(fd)
            sys.stdout.write("\x1b[?25l")
            sys.stdout.write(question + "\n")
            self._draw_options(options, selected, first=True)
            while True:
                try:
                    key = _read_key(fd)
                except KeyboardInterrupt:
                    key = "cancel"
                if key in ("up", "k"):
                    selected = (selected - 1) % len(options)
                    self._draw_options(options, selected, first=False)
                elif key in ("down", "j"):
                    selected = (selected + 1) % len(options)
                    self._draw_options(options, selected, first=False)
                elif key == "enter":
                    break
                elif key is not None and key in shortcuts:
                    selected = shortcuts[key]
                    break
                elif key in ("cancel", "escape") or key is None:
                    selected = default
                    break
            self._draw_options(options, selected, first=False)
            print(_style(f"  · selected: {options[selected]}", "2"))
            return selected
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()

    def _draw_options(
        self,
        options: list[str],
        selected: int,
        *,
        first: bool,
    ) -> None:
        if not first:
            sys.stdout.write(f"\x1b[{len(options)}A")
        for index, option in enumerate(options):
            marker = "❯" if index == selected else " "
            line = f"  {marker} {option}"
            if index == selected and _USE_COLOR:
                line = f"\x1b[7m{line}\x1b[0m"
            sys.stdout.write(f"\r\x1b[2K{line}\n")
        sys.stdout.flush()

    @staticmethod
    def _select_text(
        question: str,
        options: list[str],
        shortcuts: dict[str, int],
        default: int,
    ) -> int:
        labels = "/".join(options)
        try:
            answer = input(f"{question} [{labels}] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if answer in shortcuts:
            return shortcuts[answer]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agent_harness.chat",
        description="Interactive terminal client for the operations agent harness.",
    )
    parser.add_argument(
        "--continue",
        dest="resume_last",
        action="store_true",
        help="Resume the last conversation used in this directory.",
    )
    parser.add_argument("--conversation-id", help="Resume a specific conversation.")
    parser.add_argument("--title", help="Title for a newly created conversation.")
    return parser.parse_args(argv)


def load_saved_conversation_id() -> str | None:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    conversation_id = data.get("conversation_id")
    return conversation_id if isinstance(conversation_id, str) else None


def save_conversation_id(conversation_id: str) -> None:
    try:
        STATE_FILE.write_text(
            json.dumps({"conversation_id": conversation_id}),
            encoding="utf-8",
        )
    except OSError:
        pass


def load_input_history() -> None:
    if readline is None or not sys.stdin.isatty():
        return
    readline.set_history_length(1000)
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass


def save_input_history() -> None:
    if readline is None or not sys.stdin.isatty():
        return
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


class ChatSession:
    def __init__(
        self,
        repository: RunRepository,
        runner: AgentRunner,
        service: ConversationService,
        conversation_id: str,
        console: Console | None = None,
    ) -> None:
        self._repository = repository
        self._runner = runner
        self._service = service
        self._conversation_id = conversation_id
        self._console = console or Console()
        self._last_run_id: str | None = None

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def create_conversation(self, title: str | None) -> None:
        record = self._service.create_conversation(title)
        self._conversation_id = record.id
        self._last_run_id = None
        save_conversation_id(record.id)
        print(f"new conversation: {record.id}")

    async def send(self, content: str) -> None:
        try:
            view, replayed = await self._await_with_progress(
                self._service.send_message(
                    self._conversation_id,
                    content,
                    None,
                ),
                self._send_run_id,
            )
        except ConversationNotFoundError:
            self._console.step(
                "The saved conversation no longer exists. Use /new to start one.",
                error=True,
            )
            return
        except ConversationBusyError as exc:
            self._console.step(
                f"Conversation busy: run {exc.active_run_id} ({exc.active_status})",
                error=True,
            )
            if exc.pending_approval_id:
                self._console.step(f"pending approval: {exc.pending_approval_id}")
            return
        except ConversationTurnLimitError:
            self._console.step(
                "This conversation reached its turn limit. Use /new to start one.",
                error=True,
            )
            return

        view = await self._resolve_approval(view)
        self._last_run_id = view.id
        record = self._repository.get_run(view.id)
        self.print_result(view, record.turn_index if record else None)

    async def _resolve_approval(self, view: RunView) -> RunView:
        while view.status == "WAITING_APPROVAL" and view.pending_approval is not None:
            approval = view.pending_approval
            question = (
                "approval › "
                f"{approval.tool_name} {_compact_json(approval.arguments)}\n"
                "  Allow this action?"
            )
            choice = self._console.select(
                question,
                ["Approve", "Reject"],
                {"y": 0, "n": 1},
                default=1,
            )
            decision = "approve" if choice == 0 else "reject"
            view = await self._await_with_progress(
                self._runner.decide_approval(view.id, approval.id, decision),
                lambda result: result.id,
            )
        return view

    async def _await_with_progress(
        self,
        awaitable: Any,
        run_id_from_result: Callable[[Any], str | None],
    ) -> Any:
        task = asyncio.ensure_future(awaitable)
        seen: set[int] = set()
        live_run_id: str | None = None
        while not task.done():
            active = self._repository.get_active_conversation_turn(
                self._conversation_id
            )
            if active is not None:
                live_run_id = active.id
                self._render_new_events(active.id, seen)
            await asyncio.sleep(0.15)
        result = task.result()
        run_id = run_id_from_result(result) or live_run_id
        if run_id is not None:
            self._render_new_events(run_id, seen)
        return result

    @staticmethod
    def _send_run_id(result: tuple[RunView, bool]) -> str | None:
        view, replayed = result
        return None if replayed else view.id

    def _render_new_events(self, run_id: str, seen: set[int]) -> None:
        for event in self._repository.list_events(run_id):
            if event.sequence in seen:
                continue
            seen.add(event.sequence)
            self._console.event(event)

    def print_result(self, view: RunView, turn_index: int | None) -> None:
        if view.final_answer:
            self._console.assistant(view.final_answer)
        if view.error:
            self._console.step(
                f"{view.error.code}: {view.error.message}",
                error=True,
            )
        if view.status == "COMPLETED":
            print()
            return
        label = f"turn {turn_index}" if turn_index is not None else "turn"
        self._console.status(f"{label} · {view.status}")

    def print_history(self, limit: int) -> None:
        turns = self._service.list_turns_page(self._conversation_id, limit, None)
        if not turns:
            print("No turns yet.")
            return
        for turn in turns:
            answer = turn.final_answer or turn.error_code or ""
            print(f"turn {turn.turn_index} [{turn.status}] {turn.objective}")
            if answer:
                print(f"  {answer}")

    def print_trace(self) -> None:
        if self._last_run_id is None:
            print("No turn has run in this session yet.")
            return
        print(f"run: {self._last_run_id}")
        for event in self._repository.list_events(self._last_run_id):
            print(f"  {event.sequence:>3} {event.event_type}")


async def run_repl(
    session: ChatSession,
    console: Console,
    mode: str = "new",
) -> None:
    print("Operations agent harness. Type /help for commands, /exit to leave.")
    print(f"conversation ({mode}): {session.conversation_id}")
    while True:
        try:
            line = input(console.user_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            return
        if line == "/help":
            print(HELP_TEXT)
            continue
        if line == "/id":
            print(session.conversation_id)
            continue
        if line == "/trace":
            session.print_trace()
            continue
        if line == "/new" or line.startswith("/new "):
            title = line[len("/new") :].strip() or None
            session.create_conversation(title)
            continue
        if line == "/history" or line.startswith("/history "):
            raw_limit = line[len("/history") :].strip()
            try:
                limit = int(raw_limit) if raw_limit else 10
            except ValueError:
                print("Usage: /history [limit]")
                continue
            session.print_history(max(1, min(limit, 100)))
            continue
        if line.startswith("/"):
            print("Unknown command. Type /help.")
            continue
        await session.send(line)


async def main_async(args: argparse.Namespace) -> int:
    settings = load_settings()
    database = Database(settings.database_url)
    database.initialize()
    repository = RunRepository(database)
    repository.mark_interrupted_runs()
    llm = KimiAdapter(
        api_key=settings.kimi_api_key.get_secret_value(),
        model=settings.kimi_model,
        base_url=str(settings.kimi_base_url),
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_base_seconds=settings.retry_backoff_base_seconds,
        retry_backoff_max_seconds=settings.retry_backoff_max_seconds,
    )
    registry = ToolRegistry.from_default_files()
    runner = AgentRunner(repository, llm, settings, registry=registry)
    service = ConversationService(repository, llm, settings, runner)

    try:
        conversation_id = args.conversation_id
        mode = "resumed" if args.conversation_id else "new"
        if conversation_id is None and args.resume_last:
            conversation_id = load_saved_conversation_id()
            mode = "resumed"
            if conversation_id is None:
                print("No saved conversation found; starting a new one.")
                mode = "new"
        if conversation_id is not None:
            try:
                service.require_conversation(conversation_id)
            except ConversationNotFoundError:
                print(f"Conversation {conversation_id} was not found; starting a new one.")
                conversation_id = None
                mode = "new"
        if conversation_id is None:
            conversation_id = service.create_conversation(args.title).id
        save_conversation_id(conversation_id)
        load_input_history()
        console = Console()
        session = ChatSession(repository, runner, service, conversation_id, console)
        await run_repl(session, console, mode)
        return 0
    finally:
        save_input_history()
        await llm.aclose()
        database.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
