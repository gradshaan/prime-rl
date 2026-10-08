SYSTEM_PROMPT = """\
Solve the physics problem carefully. Show the reasoning needed to explain your
physical model, equations, assumptions, units, and checks. Put every requested
final output exactly once in a single JSON array between <final> and </final>.
Each entry must have exactly these fields: label, value, unit. Values and units
must be strings, or unit may be null for a dimensionless answer. Use the output
labels given in the question. Example:
<final>
[{"label":"a.acceleration","value":"-4.27","unit":"m/s^2"},
 {"label":"a.tension","value":"12.67","unit":"N"}]
</final>
Do not put guesses, intermediate values, or duplicate outputs in the final block.
"""


def task_prompt(shared_context: str, question: str, labels: list[str]) -> str:
    body = "\n\n".join(part for part in [shared_context, question] if part).strip()
    return f"Output exactly these labeled quantities: {', '.join(labels)}.\n\n{body}"
