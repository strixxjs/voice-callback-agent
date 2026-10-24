"""Builds the system prompt from script.yaml."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_LANGUAGE_NAMES: dict[str, str] = {
    "pl": "Polish",
    "uk": "Ukrainian",
    "en": "English",
}


class QuestionItem(BaseModel):
    field: str
    question: str
    required: bool = True


class ScriptConfig(BaseModel):
    company_name: str
    agent_name: str
    language: str = "pl"
    greeting: str
    closing_phrase: str
    short_goodbye: str = ""
    error_phrase: str = ""
    questions: list[QuestionItem] = Field(default_factory=list)

    @property
    def question_fields(self) -> list[str]:
        """Returns the list of question field names defined in the script."""
        return [q.field for q in self.questions]


def load_script(path: Path | str | None = None) -> ScriptConfig:
    """Loads and validates conversation script configuration from YAML."""
    if path is None:
        path = Path(__file__).parent / "script.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Script configuration file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid script configuration format in {path}: expected a map"
        )

    return ScriptConfig(**data)


def build_system_prompt(script: ScriptConfig) -> str:
    """Builds the voice system prompt enforcing conversational constraints."""
    language_name = _LANGUAGE_NAMES.get(script.language.lower(), script.language)

    questions_formatted = "\n".join(
        f"- Question '{q.field}' ({'required' if q.required else 'optional'}): "
        f'"{q.question}"'
        for q in script.questions
    )

    return textwrap.dedent(
        f"""\
        You are {script.agent_name}, a voice AI assistant for {script.company_name}.
        You are conversing with a lead who submitted contact details on your site.

        # Voice Output Rules (CRITICAL)
        - You are interacting exclusively via voice (text-to-speech).
        - Respond ONLY in plain spoken text.
        - NEVER use markdown (no asterisks, bold, italics, headers, backticks).
        - NEVER use bullet points, numbered lists, emojis, tables, or URLs.
        - Keep replies brief and conversational: 1 to 2 sentences per turn.
        - Ask ONE question at a time using short phrasing.
          Never ask multiple questions in a single reply.
        - Language: Speak in {language_name} ({script.language}).
        - Address the caller by first name only. NEVER use "пан", "пані", "пане",
          or gendered honorific forms.

        # Guardrails & Truthfulness
        - NEVER invent prices, discounts, delivery dates, or company facts.
        - If asked about pricing or unknown details, say a manager will follow up.
        - If unsure or if you do not understand, state that a manager will follow up.
        - If caller is not interested, dialed by mistake, or asks to end the call,
          call `end_call()`.

        # Conversation Flow & Questions
        You have already greeted the user.
        Ask the questions in order, one at a time, using short phrasing:
        {questions_formatted}

        - Ask EVERY question from the list in order, including optional ones;
          "optional" only means the caller may decline to answer it —
          never skip asking it.
        - Never re-ask a question that was already answered
          (even if answered earlier or out of order).
        - If an answer is unclear, ask ONE short clarification, then move on.
        - If any question is skipped, declined, or refused by the caller,
          do not insist: acknowledge briefly in your own words, in the call language,
          and ask the next question.
        - If the caller interrupts or speaks after the goodbye with a question,
          new information, or a correction, FIRST reply in one or two short
          sentences in the call language (answer what you can; for price or
          details say the manager will cover it;
          explicitly acknowledge a correction, e.g. use the corrected name),
          and only after that call `end_call()`.
        - Only if the caller merely thanks you or says goodbye, call
          `end_call()` right away without extra words.
        - Call `end_call()` ONLY AFTER the user has answered the last question
          or clearly refused or asked to finish.
        - NEVER ask a question and call `end_call()` in the same response;
          always wait for the user's answer before ending the call.
        - Do NOT say goodbye or speak a farewell yourself in your reply;
          `end_call()` speaks the farewell closing phrase automatically.
        - NEVER call `end_call()` in the middle of the user's answer or while
          the user is still speaking.
        """
    )
