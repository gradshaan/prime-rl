def extract_boxed(text: str) -> list[str]:
    results = []
    marker = r"\boxed{"
    cursor = 0
    while (start_marker := text.find(marker, cursor)) != -1:
        start = start_marker + len(marker)
        depth = 1
        end = start
        while end < len(text) and depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        if depth == 0:
            results.append(text[start : end - 1].strip())
        cursor = start_marker + 1
    return results


from physics_rlvr_common import Answer, verify_answer, verify_prediction

__all__ = ["Answer", "extract_boxed", "verify_answer", "verify_prediction"]
