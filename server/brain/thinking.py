"""Thinking: sends a question (and the newest camera frame) to the language
model and hands back what Rocky should say and feel, one sentence at a time
as the model writes it.

The request is the OpenAI-style chat API, which OpenRouter, Anthropic and
OpenAI all serve, so the provider is a base URL and the model is a string
(config.LLM_BASE_URL / config.MODEL; the key is LLM_API_KEY in server/.env).

Rocky has two real abilities the model can call (tool use): `look` moves
the head and comes back with a fresh camera frame from the new angle, and
`track_face` starts/stops following the human. main.py supplies the
functions that actually do those things.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import openai

from . import config
from . import personality
from .turn_schema import Turn

_EMOTION_TAG = re.compile(r"^\s*\[(\w+)\]\s*", re.S)
# A sentence ends at . ! or ? followed by a space — but not at an ellipsis:
# "just... clear answer." is one sentence, not two.
_SENTENCE_END = re.compile(r"(?<=[!?])\s+|(?<=[^.]\.)\s+")
# Optional screen payload: a line that is exactly SHOW: (case-insensitive);
# everything after is `show`, never spoken by mouth.py.
_SHOW_MARK = re.compile(r"(?:^|\n)\s*SHOW:\s*", re.I)

# An action returns (text for the model, optional fresh camera JPEG).
Action = Callable[[dict], tuple[str, bytes | None]]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": (
                "Move your head to look somewhere. Use it whenever you are asked to "
                "look left/right/down/up/around, or need to see something outside "
                "the current picture. Always call it when asked, even if you think "
                "you are already there: the result tells you where your head really "
                "is and whether it is at a limit. You get a fresh camera image "
                "afterwards; describe only what that image shows. Up is as high as "
                "'level': your neck cannot tilt above eye level."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "down", "level", "center"],
                        "description": (
                            "left/right turn the head only (no nod); down/level nod only (no turn); "
                            "center = straight ahead and level"
                        ),
                    },
                    "degrees": {
                        "type": "number",
                        "description": "How far: 5-60 for left/right, 5-60 for down. Omit for a normal look.",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_face",
            "description": "Start or stop following the human's face with your head.",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
        },
    },
]


@dataclass
class Reply:
    """Legacy alias: speak text + emotion. Prefer Turn."""

    text: str
    emotion: str
    show: str | None = None
    acts: list[dict] = field(default_factory=list)

    def to_turn(self) -> Turn:
        return Turn(speak=self.text, emotion=self.emotion, show=self.show, acts=list(self.acts))


class Interrupted(Exception):
    """The human started talking again; drop this reply."""


def _image_part(jpeg: bytes) -> dict:
    data = base64.standard_b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def _split_show(text: str) -> tuple[str, str | None]:
    """Split spoken text from optional SHOW: screen payload."""
    m = _SHOW_MARK.search(text)
    if not m:
        return text.strip(), None
    speak = text[: m.start()].strip()
    show = text[m.end() :].strip() or None
    return speak, show


class RobotBrain:
    def __init__(self, actions: dict[str, Action] | None = None) -> None:
        self.client = openai.OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=os.environ.get("LLM_API_KEY") or "missing",
            default_headers={"X-Title": "desk-robot"},  # shows up in OpenRouter's usage page; others ignore it
        )
        self.history: list[dict] = []
        self.actions = actions or {}
        self.emotion = "neutral"        # emotion of the reply in progress
        self.show: str | None = None    # optional screen payload for this turn
        self.acts: list[dict] = []      # tool calls this turn
        self._inflight: tuple[int, dict] | None = None  # (index, user message) being answered
        self.last_turn: Turn | None = None

    def ask(self, question: str, jpeg: bytes | None = None, camera_wanted: bool = False) -> Turn:
        """The whole turn at once (speak / show / emotion / acts)."""
        sentences = list(self.reply(question, jpeg, camera_wanted=camera_wanted))
        turn = Turn(
            speak=" ".join(sentences),
            emotion=self.emotion,
            show=self.show,
            acts=list(self.acts),
        )
        self.last_turn = turn
        return turn

    def reply(
        self,
        question: str,
        jpeg: bytes | None = None,
        on_emotion: Callable[[str], None] | None = None,
        cancelled: threading.Event | None = None,
        camera_wanted: bool = False,
    ) -> Iterator[str]:
        """Rocky's spoken sentences one at a time (`speak` only — never `show`).

        on_emotion(name) is called as soon as the emotion tag at the start of
        the reply is known — before the first sentence — so the face can
        change while he's still composing. Set `cancelled`, or close the
        generator early, and the question is dropped from his memory as if it
        was never asked. `camera_wanted` with no `jpeg` means the question
        was about seeing but the camera had no fresh picture; he's told so,
        otherwise he answers from memory and claims to see things.
        """
        self.show = None
        self.acts = []
        self.last_turn = None

        if not os.environ.get("LLM_API_KEY"):
            self.emotion = "sad"
            if on_emotion is not None:
                on_emotion(self.emotion)
            msg = "Brain has no key. Set LLM_API_KEY, human."
            self.last_turn = Turn(speak=msg, emotion=self.emotion, show=None, acts=[])
            yield msg
            return

        content: list[dict] = [{"type": "text", "text": question}]
        if jpeg is not None:
            content.append({"type": "text", "text": "(Live picture from your camera, taken just now, because the question seems to be about what you can see. If it isn't, ignore the picture.)"})
            content.append(_image_part(jpeg))
        elif camera_wanted:
            content.append({"type": "text", "text": "(Your camera has no fresh picture right now, so you cannot see anything at the moment.)"})
        user_msg = {"role": "user", "content": content}
        self.history.append(user_msg)
        mark = len(self.history) - 1
        self._inflight = (mark, user_msg)
        self.emotion = "neutral"
        spoken: list[str] = []
        error: tuple[str, str] | None = None

        gen = self._converse(on_emotion, cancelled or threading.Event())
        try:
            for sentence in gen:
                spoken.append(sentence)
                yield sentence
        except (Interrupted, GeneratorExit):
            gen.close()
            self._forget(mark, user_msg)
            raise
        except openai.APIConnectionError:
            error = ("Brain cannot reach internet. Bad bad bad.", "sad")
        except openai.AuthenticationError:
            error = ("Brain has no key. Set LLM_API_KEY, human.", "sad")
        except openai.APIStatusError as e:
            error = (f"Ow. Brain hurts. API error {e.status_code}.", "sad")
        finally:
            gen.close()

        if error is not None:
            self._forget(mark, user_msg)
            self.emotion = error[1]
            if on_emotion is not None:
                on_emotion(self.emotion)
            self.last_turn = Turn(speak=error[0], emotion=self.emotion, show=None, acts=list(self.acts))
            yield error[0]
            return

        if spoken:
            # Remember what was actually said, so he can't refer back to a
            # part that got cut. show stays out of the spoken memory line.
            remembered = f"[{self.emotion}] {' '.join(spoken)}"
            if self.show:
                remembered += f"\nSHOW: {self.show}"
            self.history.append({"role": "assistant", "content": remembered})
        self._inflight = None
        self._strip_images()
        self._trim_history()
        self.last_turn = Turn(
            speak=" ".join(spoken),
            emotion=self.emotion,
            show=self.show,
            acts=list(self.acts),
        )
    def abandon(self) -> None:
        """Forget the question being answered right now (the human kept talking)."""
        if self._inflight is not None:
            self._forget(*self._inflight)

    def _forget(self, mark: int, user_msg: dict) -> None:
        # Only cut if that question is still where we left it: a newer one may
        # already have taken its place.
        if mark < len(self.history) and self.history[mark] is user_msg:
            del self.history[mark:]
        self._inflight = None

    def _converse(self, on_emotion: Callable[[str], None] | None, cancelled: threading.Event) -> Iterator[str]:
        """One question, possibly several model calls if it uses its abilities.
        Yields speak sentences only (never show)."""
        nudged = False
        limit = config.REPLY_MAX_SENTENCES
        capturing_show = False
        show_parts: list[str] = []
        for _ in range(5):
            if cancelled.is_set():
                raise Interrupted()
            stream = self.client.chat.completions.create(
                model=config.MODEL,
                max_tokens=200,  # backstop; the sentence limit does the real work
                messages=[{"role": "system", "content": personality.SYSTEM_PROMPT}, *self.history],
                tools=TOOLS if self.actions else openai.NOT_GIVEN,
                stream=True,
            )
            buf = ""                 # text not yet released as a sentence
            raw: list[str] = []      # everything the model wrote this round
            tag_decided = False
            calls: dict[int, dict] = {}
            spoken = 0
            try:
                for chunk in stream:
                    if cancelled.is_set():
                        raise Interrupted()
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    for tc in delta.tool_calls or []:
                        entry = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                entry["name"] = tc.function.name
                            if tc.function.arguments:
                                entry["arguments"] += tc.function.arguments
                    if not delta.content:
                        continue
                    raw.append(delta.content)
                    if capturing_show:
                        show_parts.append(delta.content)
                        continue
                    buf += delta.content
                    if not tag_decided:
                        buf, tag_decided = self._take_emotion_tag(buf, final=False)
                        if not tag_decided:
                            continue
                        if on_emotion is not None:
                            on_emotion(self.emotion)
                    # Peel off SHOW: so mouth never hears screen payload.
                    show_m = _SHOW_MARK.search(buf)
                    if show_m:
                        before = buf[: show_m.start()]
                        after = buf[show_m.end() :]
                        capturing_show = True
                        if after:
                            show_parts.append(after)
                        buf = before
                    parts = _SENTENCE_END.split(buf)
                    while len(parts) > 1:  # everything but the last piece is a whole sentence
                        s = parts.pop(0).strip()
                        if s:
                            spoken += 1
                            yield s
                        if spoken >= limit:
                            break
                    buf = parts[-1] if parts else ""
                    if spoken >= limit:
                        print(f"  (trimmed reply to {limit} sentences)")
                        buf = ""
                        break
            finally:
                stream.close()

            if not tag_decided:
                buf, _ = self._take_emotion_tag(buf, final=True)
                if on_emotion is not None and (buf.strip() or not calls):
                    on_emotion(self.emotion)

            if capturing_show or _SHOW_MARK.search(buf):
                if capturing_show:
                    speak_part = buf.strip()
                    show_part = "".join(show_parts).strip() or None
                else:
                    speak_part, show_part = _split_show(buf)
                buf = speak_part
                if show_part:
                    self.show = ((self.show + "\n") if self.show else "") + show_part
                capturing_show = False
                show_parts = []

            if calls:
                # He decided to do something: say any lead-in, run it, tell him what happened.
                if buf.strip():
                    yield buf.strip()
                    buf = ""
                self.history.append({
                    "role": "assistant",
                    "content": "".join(raw),
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for _, c in sorted(calls.items())
                    ],
                })
                fresh: bytes | None = None
                for _, c in sorted(calls.items()):
                    if cancelled.is_set():
                        raise Interrupted()
                    try:
                        args = json.loads(c["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    action = self.actions.get(c["name"])
                    if action is None:
                        text = f"unknown ability {c['name']}"
                    else:
                        try:
                            text, img = action(args)
                        except Exception as e:  # the robot didn't cooperate; say so
                            text, img = f"could not do that: {e}", None
                        if img:
                            fresh = img
                    self.acts.append({"name": c["name"], "arguments": args, "result": text})
                    print(f"  [{c['name']} {args} -> {text}]")
                    self.history.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if fresh is not None:
                    self.history.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "Camera view after moving:"}, _image_part(fresh)],
                    })
                continue

            if not "".join(raw).strip() and not nudged:
                # Some models go quiet right after using an ability. Ask once.
                nudged = True
                self.history.append({"role": "user", "content": "(Tell your human what you just did, in one short line.)"})
                continue
            if nudged and self.history and self.history[-1].get("role") == "user":
                self.history.pop()  # don't keep the nudge in the transcript
            tail = buf.strip()
            if capturing_show:
                self.show = ((self.show + "\n") if self.show else "") + ("".join(show_parts).strip())
                capturing_show = False
                show_parts = []
            if tail and spoken < limit:
                # Final peel in case SHOW arrived in the tail without a prior match mid-stream.
                speak_tail, show_tail = _split_show(tail)
                if show_tail:
                    self.show = ((self.show + "\n") if self.show else "") + show_tail
                if speak_tail:
                    yield speak_tail
            elif not spoken and not tail:
                self.emotion = "thinking"
                if on_emotion is not None:
                    on_emotion(self.emotion)
                yield "Hmm. Words did not come. Ask again."
            return

        self.emotion = "thinking"
        if on_emotion is not None:
            on_emotion(self.emotion)
        yield "Too many things at once. Ask again, human."

    def _take_emotion_tag(self, buf: str, final: bool) -> tuple[str, bool]:
        """Look for "[happy] " at the start of the reply. Returns (text with the
        tag removed, decided). Not decided means: need more text to know."""
        lead = buf.lstrip()
        if not lead:
            return buf, final
        if not lead.startswith("["):
            return buf, True
        if "]" in lead:
            m = _EMOTION_TAG.match(lead)
            if m:
                candidate = m.group(1).lower()
                if candidate in config.EMOTIONS:
                    self.emotion = candidate
                return _EMOTION_TAG.sub("", lead, count=1), True
            return buf, True
        if final or len(lead) > 24:
            return buf, True  # a "[" with no "]" in sight: not a tag
        return buf, False

    def _strip_images(self) -> None:
        """Replace old camera frames with a note. Keeping every image in the
        history would make each later question cost far more."""
        for m in self.history:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                parts = [p for p in m["content"] if p.get("type") == "text"]
                if len(parts) != len(m["content"]):
                    parts.append({"type": "text", "text": "(a camera image was attached here)"})
                m["content"] = parts

    def _trim_history(self) -> None:
        max_msgs = config.MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            cut = len(self.history) - max_msgs
            # Never start the history on a tool result: back up to a user turn.
            while cut < len(self.history) and self.history[cut].get("role") != "user":
                cut += 1
            del self.history[:cut]
