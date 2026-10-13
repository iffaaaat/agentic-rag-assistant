import json

from src.config import CHAT_MODEL, client


def ask_for_json(prompt):
    """Request a structured JSON response from the LLM."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)
