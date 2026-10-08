from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import pint
import sympy
from sympy import Rational, simplify
from sympy.parsing.latex import parse_latex
from sympy.printing.latex import LatexPrinter

_UNIT_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
_FINAL_OPEN = "<final>"
_FINAL_CLOSE = "</final>"
_LATEX_TEXT = re.compile(r"\\(?:text|mathrm|mbox)\{([^{}]*)\}")


class _ArithmeticLatexPrinter(LatexPrinter):
    def _print_exp(self, expression):
        return r"\exp(" + self._print(expression.args[0]) + ")"


@dataclass(frozen=True)
class Answer:
    value: str
    unit: str | None
    answer_type: str
    tolerance: float | str | None = None
    verifier: str = "string"
    equivalent_forms: list[str] = field(default_factory=list)
    subproblem_id: str | None = None
    label: str | None = None
    atol: float | None = None
    rtol: float | None = None
    assumptions: list[str] = field(default_factory=list)
    bindings: dict[str, str] = field(default_factory=dict)

    @property
    def output_label(self) -> str | None:
        return self.label or self.subproblem_id


def extract_final_json(completion: str) -> list[dict[str, Any]] | None:
    if completion.count(_FINAL_OPEN) != 1 or completion.count(_FINAL_CLOSE) != 1:
        return None
    start = completion.find(_FINAL_OPEN) + len(_FINAL_OPEN)
    end = completion.find(_FINAL_CLOSE)
    if end < start:
        return None

    try:
        payload = json.loads(
            completion[start:end].strip(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, list) or not payload:
        return None

    predictions: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict) or set(entry) != {"label", "value", "unit"}:
            return None
        if not isinstance(entry["label"], str) or not entry["label"].strip():
            return None
        if not isinstance(entry["value"], str) or not entry["value"].strip():
            return None
        if len(entry["label"]) > 100 or len(entry["value"]) > 1000:
            return None
        if entry["unit"] is not None and not isinstance(entry["unit"], str):
            return None
        if isinstance(entry["unit"], str) and len(entry["unit"]) > 100:
            return None
        predictions.append(entry)

    labels = [entry["label"] for entry in predictions]
    if len(labels) != len(set(labels)):
        return None
    return predictions


def verify_prediction(completion: str, answers: list[Answer]) -> float:
    if not answers or any(validate_answer(answer, require_label=True) for answer in answers):
        return 0.0
    labels = [answer.output_label for answer in answers]
    if len(labels) != len(set(labels)):
        return 0.0

    predictions = extract_final_json(completion)
    if predictions is None:
        return 0.0
    by_label = {prediction["label"]: prediction for prediction in predictions}
    expected_labels = set(labels)
    if not set(by_label).issubset(expected_labels):
        return 0.0

    correct = sum(
        answer.output_label in by_label and verify_answer(by_label[answer.output_label], answer)  # type: ignore[index]
        for answer in answers
    )
    return correct / len(answers)


def verify_answer(prediction: dict[str, Any] | str, answer: Answer | dict[str, Any]) -> bool:
    expected = answer_from_dict(answer) if isinstance(answer, dict) else answer
    if validate_answer(expected):
        return False
    if isinstance(prediction, str):
        predicted_value, predicted_unit = _split_latex_quantity(prediction)
    else:
        if not isinstance(prediction.get("value"), str):
            return False
        if prediction.get("unit") is not None and not isinstance(prediction.get("unit"), str):
            return False
        predicted_value = prediction["value"].strip()
        predicted_unit = prediction["unit"]

    expected_values = [expected.value, *expected.equivalent_forms]
    if expected.answer_type in {"numeric", "numerical"} or expected.verifier == "numeric":
        return any(
            _numeric_match(predicted_value, predicted_unit, value, expected)
            for value in expected_values
        )
    if expected.answer_type in {"symbolic", "expression"} or expected.verifier in {"sympy", "expression"}:
        try:
            # An expression in physical symbols carries its unit implicitly, so a missing unit on either side is not a mismatch.
            transform = ((sympy.S.One, sympy.S.Zero) if predicted_unit is None or expected.unit is None
                         else _symbolic_unit_transform(predicted_unit, expected.unit))
        except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError, ValueError, TypeError):
            return False
        if transform is None:
            return False
        scale, offset = transform
        return any(_symbolic_match(predicted_value, value, scale, expected.assumptions, offset,
                                   expected.bindings, expected.rtol or 0) for value in expected_values)
    return _normalize_text(predicted_value) == _normalize_text(expected.value) and predicted_unit == expected.unit


def answer_from_dict(raw: dict[str, Any]) -> Answer:
    value = raw.get("value", "")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("answer value must be a string or number")
    unit = raw.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError("answer unit must be a string or null")
    tolerance = raw.get("tolerance")
    if tolerance is not None and not isinstance(tolerance, (int, float, str)):
        raise ValueError("answer tolerance must be numeric")
    equivalent_forms = raw.get("equivalent_forms", [])
    assumptions = raw.get("assumptions", [])
    if not isinstance(equivalent_forms, list) or not all(isinstance(value, str) for value in equivalent_forms):
        raise ValueError("equivalent_forms must be a list of strings")
    if not isinstance(assumptions, list) or not all(isinstance(value, str) for value in assumptions):
        raise ValueError("assumptions must be a list of strings")
    label = raw.get("label") or raw.get("output_label")
    subproblem_id = raw.get("subproblem_id")
    if label is not None and not isinstance(label, str):
        raise ValueError("answer label must be a string")
    if subproblem_id is not None and not isinstance(subproblem_id, str):
        raise ValueError("subproblem_id must be a string")
    return Answer(
        value=str(value),
        unit=unit,
        answer_type=str(raw.get("answer_type", "string")),
        tolerance=tolerance,
        verifier=str(raw.get("verifier", "string")),
        equivalent_forms=equivalent_forms,
        subproblem_id=subproblem_id,
        label=label,
        atol=raw.get("atol"),
        rtol=raw.get("rtol"),
        assumptions=assumptions,
        bindings=raw.get("bindings", {}),
    )


def validate_answer(answer: Answer, *, require_label: bool = False) -> list[str]:
    errors = []
    if not isinstance(answer.value, str) or not answer.value.strip():
        errors.append("answer value is empty")
    if answer.unit is not None and not isinstance(answer.unit, str):
        errors.append("answer unit must be a string or null")
    if require_label and (not isinstance(answer.output_label, str) or not answer.output_label.strip()):
        errors.append("answer label is empty")
    if answer.answer_type in {"numeric", "numerical"}:
        if answer.verifier != "numeric":
            errors.append("numeric answer must use numeric verifier")
        raw_rtol = answer.rtol if answer.rtol is not None else answer.tolerance
        if raw_rtol is None:
            errors.append("numeric answer needs an explicit relative tolerance")
        elif not _valid_tolerance(raw_rtol):
            errors.append("relative tolerance must be finite and non-negative")
        if answer.atol is None or not _valid_tolerance(answer.atol):
            errors.append("absolute tolerance must be finite and non-negative")
        if isinstance(answer.value, str) and answer.value.strip() and _parse_number(answer.value) is None:
            errors.append("numeric answer value is not a supported numeric expression")
    elif answer.answer_type in {"symbolic", "expression"}:
        if answer.verifier not in {"sympy", "expression"}:
            errors.append("symbolic answer must use sympy or expression verifier")
        if isinstance(answer.value, str) and answer.value.strip() and not _is_parseable_symbolic(answer.value):
            errors.append("symbolic answer value is not a supported expression")
    else:
        errors.append(f"unsupported answer type {answer.answer_type!r}")
    if not isinstance(answer.equivalent_forms, list) or not all(
        isinstance(value, str) for value in answer.equivalent_forms
    ):
        errors.append("equivalent_forms must be a list of strings")
    if not isinstance(answer.assumptions, list) or not all(
        isinstance(value, str) for value in answer.assumptions
    ):
        errors.append("assumptions must be a list of strings")
    else:
        try:
            _domain_symbols(answer.assumptions)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        _numeric_bindings(answer.bindings)
    except (TypeError, ValueError):
        errors.append("bindings must map single symbols to finite numeric values")
    if isinstance(answer.unit, str) and answer.unit:
        normalized_unit = _normalize_unit(answer.unit)
        if normalized_unit is not None:
            try:
                _UNIT_REGISTRY.Quantity(1, normalized_unit)
            except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError, TypeError, ValueError):
                errors.append(f"unit {answer.unit!r} is not recognized")
    return errors


def _valid_tolerance(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0


def _numeric_match(
    prediction: str,
    prediction_unit: str | None,
    expected_text: str,
    answer: Answer,
) -> bool:
    expected_number = _parse_number(expected_text)
    prediction_number = _parse_number(prediction)
    if expected_number is None or prediction_number is None:
        return False

    try:
        converted_prediction = _units_match(
            prediction_unit,
            answer.unit,
            prediction_number,
            expected_number,
        )
    except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError, ValueError, TypeError):
        return False
    if converted_prediction is None:
        return False

    rtol = answer.rtol
    if rtol is None and answer.tolerance not in (None, ""):
        rtol = float(answer.tolerance)
    if rtol is None or not math.isfinite(float(rtol)) or float(rtol) < 0:
        return False
    atol = answer.atol
    if atol is None or not math.isfinite(float(atol)) or float(atol) < 0:
        return False
    return math.isclose(
        converted_prediction,
        expected_number,
        rel_tol=float(rtol),
        abs_tol=float(atol),
    )


def _units_match(
    prediction_unit: str | None,
    expected_unit: str | None,
    prediction_number: float,
    expected_number: float,
) -> float | None:
    normalized_prediction_unit = _normalize_unit(prediction_unit)
    normalized_expected_unit = _normalize_unit(expected_unit)
    if normalized_prediction_unit is None and normalized_expected_unit is None:
        return prediction_number
    # A missing unit means a pure number, which still converts to dimensionless units such as percent.
    normalized_prediction_unit = normalized_prediction_unit or "dimensionless"
    normalized_expected_unit = normalized_expected_unit or "dimensionless"
    prediction_quantity = _UNIT_REGISTRY.Quantity(prediction_number, normalized_prediction_unit)
    expected_quantity = _UNIT_REGISTRY.Quantity(expected_number, normalized_expected_unit)
    if prediction_quantity.dimensionality != expected_quantity.dimensionality:
        return None
    return float(prediction_quantity.to(normalized_expected_unit).magnitude)


def _parse_number(text: str) -> float | None:
    cleaned = _strip_wrappers(text)
    if len(cleaned) > 512:
        return None
    cleaned = cleaned.replace(r"\times", "*").replace("×", "*").replace(r"\cdot", "*")
    cleaned = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", cleaned)
    cleaned = cleaned.replace("^", "**")
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = float(_evaluate_numeric_node(tree.body))
    except (TypeError, ValueError, SyntaxError, OverflowError, ZeroDivisionError):
        try:
            result = float(_parse_symbolic_expression(text).evalf())
        except Exception:
            return None
    return result if math.isfinite(result) else None


def _evaluate_numeric_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_numeric_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise ValueError("expression is not numeric arithmetic")

    left = _evaluate_numeric_node(node.left)
    right = _evaluate_numeric_node(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Pow):
        result = left**right
        if isinstance(result, complex):
            raise ValueError("complex numeric answers are unsupported")
        return result
    raise ValueError("unsupported numeric operator")


def normalize_expression(value: str) -> str:
    """Convert explicit arithmetic to LaTeX without evaluating Python code."""
    value = _strip_wrappers(value)
    greek_names = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
                   "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda",
                   "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma",
                   "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega"}
    # A standalone letter such as 'ω_E1' becomes the plain name 'omega_E1'; 'πr' keeps LaTeX's implicit product.
    value = "".join(
        character if character not in greek_names
        else greek_names[character] if not value[i - 1:i].isalnum() and not value[i + 1:i + 2].isalnum()
        else "\\" + greek_names[character] + " "
        for i, character in enumerate(value)
    )
    if len(value) > 1000:
        raise ValueError("expression exceeds the supported length")
    if "\\" not in value and re.fullmatch(r"[\w\s+*/().^\-]+", value):
        try:
            tree = ast.parse(value.replace("^", "**"), mode="eval")
        except SyntaxError:
            # Implicit products such as '100 x' already use LaTeX notation.
            return value
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("expression exceeds the supported complexity")
        return _ArithmeticLatexPrinter().doprint(_symbolic_node(tree.body))
    # A LaTeX expression can contain powers from the model's Python notation.
    exponent = r"\*\*\s*(\([^()]*\)|[+-]?\d+(?:\.\d+)?|[A-Za-z](?:_[A-Za-z0-9]+)?)"
    value = re.sub(exponent, lambda match: "^{" + match[1] + "}", value)
    if "**" in value:
        raise ValueError("unsupported mixed-notation exponent")
    return value


def _symbolic_node(node: ast.expr):
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        if not math.isfinite(node.value):
            raise ValueError("non-finite expression constant")
        return Rational(str(node.value))
    if isinstance(node, ast.Name):
        return sympy.Symbol(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _symbolic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _symbolic_node(node.left), _symbolic_node(node.right)
        operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                      ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                      ast.Pow: lambda: left ** right}
        operation = operations.get(type(node.op))
        if operation is not None:
            if isinstance(node.op, ast.Pow) and right.is_number and abs(right) > 1000:
                raise ValueError("exponent exceeds the supported magnitude")
            return operation()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords and len(node.args) == 1:
        functions = {name: getattr(sympy, name) for name in [
            "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log"]}
        functions.update(abs=sympy.Abs, ln=sympy.log)
        if node.func.id in functions:
            return functions[node.func.id](_symbolic_node(node.args[0]))
    raise ValueError("unsupported arithmetic syntax")


def _domain_symbols(assumptions: list[str]) -> dict:
    replacements = {}
    for assumption in assumptions:
        match = re.fullmatch(r"\s*(.+?)\s*(>|>=|<|<=)\s*0\s*", assumption)
        if not match:
            continue
        variable = parse_latex(_explicit_products(normalize_expression(match[1])), strict=True)
        if not isinstance(variable, sympy.Symbol):
            raise ValueError("domain assumption must refer to one symbol")
        predicate = {">": "positive", ">=": "nonnegative", "<": "negative", "<=": "nonpositive"}[match[2]]
        constrained = sympy.Symbol(variable.name, **{predicate: True})
        if variable in replacements and replacements[variable] != constrained:
            previous = replacements[variable]
            if previous.is_positive and predicate == "nonnegative" or previous.is_negative and predicate == "nonpositive":
                continue
            if previous.is_nonnegative and predicate == "positive" or previous.is_nonpositive and predicate == "negative":
                replacements[variable] = constrained
                continue
            raise ValueError("conflicting domain assumptions")
        replacements[variable] = constrained
    return replacements


def _symbolic_unit_transform(prediction_unit: str | None, expected_unit: str | None):
    predicted, expected = _normalize_unit(prediction_unit), _normalize_unit(expected_unit)
    if predicted is None or expected is None:
        return (sympy.S.One, sympy.S.Zero) if predicted is expected else None
    predicted_unit, expected_unit = _UNIT_REGISTRY.Unit(predicted), _UNIT_REGISTRY.Unit(expected)
    if predicted_unit.dimensionality != expected_unit.dimensionality:
        return None
    temperatures = {"kelvin": (sympy.S.One, sympy.S.Zero),
                    "degree_Celsius": (sympy.S.One, Rational(27315, 100)),
                    "degree_Fahrenheit": (Rational(5, 9), Rational(45967, 180)),
                    "degree_Rankine": (Rational(5, 9), sympy.S.Zero)}
    if str(predicted_unit) in temperatures and str(expected_unit) in temperatures:
        pscale, poffset = temperatures[str(predicted_unit)]
        escale, eoffset = temperatures[str(expected_unit)]
        return pscale / escale, (poffset - eoffset) / escale
    if (str(predicted_unit), str(expected_unit)) == ("degree", "radian"):
        return sympy.pi / 180, sympy.S.Zero
    if (str(predicted_unit), str(expected_unit)) == ("radian", "degree"):
        return 180 / sympy.pi, sympy.S.Zero
    zero, one, two = [_UNIT_REGISTRY.Quantity(value, predicted_unit).to(expected_unit).magnitude for value in [0, 1, 2]]
    if not math.isclose(two, 2 * (one - zero) + zero, rel_tol=1e-12, abs_tol=1e-12):
        return None
    return Rational(str(one - zero)), Rational(str(zero))


def _parse_symbolic_expression(value: str):
    normalized = normalize_expression(value).replace(r"\left", "").replace(r"\right", "")
    expression = parse_latex(_explicit_products(normalized), strict=True)
    return expression.xreplace({sympy.Symbol("pi"): sympy.pi})


def _numeric_bindings(bindings: dict[str, str]) -> dict:
    if not isinstance(bindings, dict):
        raise ValueError("bindings must be a dictionary")
    result = {}
    for name, value in bindings.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("binding names and values must be strings")
        symbol = _parse_symbolic_expression(name)
        number = _parse_symbolic_expression(value)
        if not isinstance(symbol, sympy.Symbol) or number.free_symbols or _parse_number(value) is None:
            raise ValueError("unsupported numeric binding")
        result[symbol] = simplify(number)
    return result


def _symbolic_match(prediction: str, expected: str, scale=1,
                    assumptions: list[str] | None = None, offset=0,
                    bindings: dict[str, str] | None = None, rtol: float = 0) -> bool:
    if len(prediction) > 1000 or len(expected) > 1000:
        return False
    try:
        predicted_expression = _parse_symbolic_expression(prediction)
        expected_expression = _parse_symbolic_expression(expected)
        bindings_map = _numeric_bindings(bindings or {})
        predicted_expression = predicted_expression.subs(bindings_map)
        expected_expression = expected_expression.subs(bindings_map)
        replacements = _domain_symbols(assumptions or [])
        predicted_expression = predicted_expression.xreplace(replacements)
        expected_expression = expected_expression.xreplace(replacements)
    except Exception:
        return False
    try:
        difference = simplify(scale * predicted_expression + offset - expected_expression)
        if difference == 0:
            return True
        if bindings_map and _valid_tolerance(rtol) and rtol > 0 and expected_expression != 0:
            relative_error = simplify(difference / expected_expression)
            if not relative_error.free_symbols:
                return abs(float(relative_error)) <= rtol
        return False
    except (TypeError, ValueError, NotImplementedError):
        return False


def _is_parseable_symbolic(value: str) -> bool:
    try:
        _parse_symbolic_expression(value)
    except Exception:
        return False
    return True


def _explicit_products(value: str) -> str:
    # Physics uses single-letter symbols; d q must not become a differential dq.
    pattern = re.compile(
        r"\\(?:text|mathrm|operatorname)\{[^{}]*\}|\\[A-Za-z]+(?:_\{[^{}]*\}|_[A-Za-z0-9])?"
        r"|[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?|[^\s]"
    )
    tokens = list(pattern.finditer(value))
    greek = {"alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "theta", "vartheta",
             "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "varphi", "chi", "psi", "omega",
             "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega"}

    def variable(token: str) -> bool:
        if token.startswith("\\"):
            return token[1:].split("_", 1)[0] in greek
        return token[0].isalpha()

    insertions = [match.end() for match, following in zip(tokens, tokens[1:])
                  if variable(match.group()) and (variable(following.group()) or following.group() == "(")]
    for offset in reversed(insertions):
        value = value[:offset] + r" \cdot " + value[offset:]
    return re.sub(r"(?<![A-Za-z\\])d\s*\\cdot", lambda _: r"(d) \cdot", value)


def _split_latex_quantity(value: str) -> tuple[str, str | None]:
    match = _LATEX_TEXT.search(value)
    if match is None:
        return value.strip(), None
    return value[: match.start()].strip(), _normalize_unit(match.group(1))


def _strip_wrappers(value: str) -> str:
    value = value.strip().strip("$")
    while value.startswith(r"\boxed{"):
        depth = 1
        end = len(r"\boxed{")
        while end < len(value) and depth:
            if value[end] == "{":
                depth += 1
            elif value[end] == "}":
                depth -= 1
            end += 1
        if depth or end != len(value):
            break
        value = value[len(r"\boxed{") : end - 1].strip()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\displaystyle", "")
    value = re.sub(r"\\(?:mathrm|text|mbox)\{([^{}]*)\}", r"\1", value)
    return value.strip()


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = _strip_wrappers(unit).replace(r"\,", " ").replace("\\", " ").strip()
    if normalized in {"", "1", "dimensionless"}:
        return None
    if normalized in {"dptr", "diopter", "diopters", "dioptre", "dioptres"}:
        return "1/m"
    normalized = normalized.replace("·", "*").replace("⋅", "*")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s+", "*", normalized)
    return normalized


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
