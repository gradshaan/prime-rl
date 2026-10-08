"""Source admission must use competition identity, not upload dates or instructions."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def collector():
    path = Path(__file__).resolve().parents[3] / "tools/phy_rl_collect.py"
    spec = importlib.util.spec_from_file_location("phy_rl_collect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_uses_competition_year_and_excludes_recent_material(collector):
    html = '''<h2>NBPhO 2003 – 2025</h2>
    <div class="wp-block-column"><p>Competition in April 2023</p>
    <a href="/uploads/2026/NBPhO_2023.pdf">Problems in English</a>
    <a href="/uploads/2026/NBPhO_2023_sol.pdf">Solutions in English</a></div>
    <div class="wp-block-column"><p>Competition in April 2025</p>
    <a href="/uploads/2025/NBPhO_2025.pdf">Problems in English</a>
    <a href="/uploads/2025/NBPhO_2025_sol.pdf">Solutions in English</a></div>'''
    rows = collector.archive_records("nbpho_archive", html)
    assert len(rows) == 1
    assert rows[0]["year"] == 2023
    assert rows[0]["problem_url"].endswith("/uploads/2026/NBPhO_2023.pdf")


def test_instruction_numbers_do_not_become_problem_identities(collector):
    paper = {"competition": "INPhO", "year": 2023, "paper_variant": "main"}
    doc = {"pages": [{"text": "1. This booklet contains 5 questions.\n"
                      "2. Non-programmable calculators are allowed.\n"
                      "Questions\n1. A particle moves in a magnetic field.\nFind its radius.\n"
                      "2. An ideal gas expands isothermally.\nFind the work.\n"}]}
    rows = collector.paper_problems(paper, doc)
    assert [r["problem_id"] for r in rows] == ["INPhO_2023_problem_1", "INPhO_2023_problem_2"]
    assert rows[0]["question_excerpt"].startswith("1. A particle")


def test_mirror_evaluation_identity_remains_original_problem(collector):
    html = '''<li data-t="thermodynamics"><span class="rtag">Theory</span>
    <span class="t">Gas</span><a class="k-problems" href="/files/ipho/IPhO_2023_Q1.pdf">Problem</a>
    <a class="k-solutions" href="/files/ipho/IPhO_2023_S1.pdf">Solution</a></li>
    <li><span class="rtag">Theory</span><a class="k-problems" href="/files/ipho/IPhO_2026_Q1.pdf">Problem</a></li>'''
    rows = collector.archive_records("ipho_olimpicos", html)
    assert len(rows) == 1
    assert rows[0]["problem_id"] == "IPhO_2023_problem_1"
    assert rows[0]["solution_url"].endswith("IPhO_2023_S1.pdf")


def test_diagram_numbers_do_not_inflate_naboj_problem_count(collector):
    paper = {"competition": "PhysicsNaboj", "year": 2023, "paper_variant": "main"}
    doc = {"pages": [{"text": "Problems\n1\nFirst physics question.\n64\nDiagram annotation.\n"
                      "2\nSecond physics question.\nSolutions\n1\nFirst solution.\n"}]}
    rows = collector.paper_problems(paper, doc)
    assert [r["problem_number"] for r in rows] == ["1", "2"]


def test_czech_school_year_and_experiment_admission(collector):
    html = '''<h3>65. ročník (2023–2024)</h3>
    <a href="archiv/65/fo65a1_z.pdf">pdf</a><a href="archiv/65/fo65a1_r.pdf">pdf</a>
    <h3>64. ročník (2022–2023)</h3>
    <a href="archiv/64/fo64a3_z.pdf">pdf</a><a href="archiv/64/fo64a3_r.pdf">pdf</a>
    <a href="archiv/64/fo64a3_pz.pdf">pdf</a><a href="archiv/64/fo64a3_pr.pdf">pdf</a>'''
    rows = collector.archive_records("czech_physics_olympiad", html)
    assert len(rows) == 1
    assert rows[0]["year"] == 2023
    assert rows[0]["paper_variant"] == "A_round3"
    assert rows[0]["problem_url"] == "https://fyzikalniolympiada.cz/archiv/64/fo64a3_z.pdf"


@pytest.fixture
def training_importer():
    path = Path(__file__).resolve().parents[3] / "tools/phy_rl_gather_training.py"
    spec = importlib.util.spec_from_file_location("phy_rl_gather_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_preserves_source_answers_and_requires_audit(training_importer):
    question = "A particle moves in a magnetic field. " * 8 + "Find its radius as a function of momentum."
    answer = r"r = \frac{p}{qB}"
    solution = "Use the Lorentz force and centripetal acceleration. " * 20
    row = training_importer.stage_record(training_importer.TEXTBOOK, 173, question, answer, solution, {})
    assert row["reference_answer"] == answer
    assert row["reference_solution"] == solution
    assert row["provenance"]["row_index"] == 173
    assert row["required_outputs"] == []
    assert row["release_status"] == "held"
    assert row["training_ready"] is False


def test_import_rejects_missing_context_and_numeric_variants(training_importer):
    question = "A particle moves as shown in the figure. " * 8 + "Find its energy E = mv^2/2."
    reasons, _ = training_importer.assess(question, "40", "Use energy conservation. " * 40)
    assert "external_context_or_image" in reasons
    assert training_importer.family_key("A 12 kg block moves at 3.5 m/s") == training_importer.family_key(
        "A 23 kg block moves at 9.2 m/s"
    )
    reasons, _ = training_importer.assess("IPhO 2026: " + question, "XX", "Use energy conservation. " * 40)
    assert "named_benchmark_or_forbidden_year" in reasons
    assert "missing_or_long_reference_answer" in reasons
    formula = r"E = \frac{m c^2}{\sqrt{1 - v^2/c^2}}"
    assert training_importer.target_is_given("Derive this formula: \\[" + formula + "\\]", "\\[" + formula + "\\]")


def test_indexed_lexical_screen_matches_exhaustive_jaccard():
    path = Path(__file__).resolve().parents[3] / "tools/phy_rl_screen.py"
    spec = importlib.util.spec_from_file_location("phy_rl_screen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    texts = ["a particle moves along a smooth track at a constant speed",
             "a particle moves along a smooth track with constant acceleration",
             "no shared long sequence here", "a particle moves along a smooth track with constant acceleration"]
    sets = [module.shingles(text) for text in texts]
    index = module.shingle_index(sets)
    for text in texts + ["entirely unrelated statement", "particle moves along a smooth track with constant acceleration"]:
        grams = module.shingles(text)
        scores = [len(grams & item) / max(1, len(grams | item)) for item in sets]
        nearest = max(range(len(scores)), key=scores.__getitem__)
        assert module.nearest_lexical(grams, index, [len(item) for item in sets]) == (nearest, scores[nearest])


@pytest.fixture
def competition_collector():
    path = Path(__file__).resolve().parents[3] / "tools/phy_rl_collect_competition.py"
    spec = importlib.util.spec_from_file_location("phy_rl_collect_competition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_page_keeps_inline_answer_out_of_question(competition_collector):
    parser = competition_collector.SolutionPage()
    parser.feed(r'<h3>Условие:</h3><p>Find speed \(v\).</p><img src="diagram.svg">'
                r'<h3>Решение:</h3><p>Use \(v=s/t\).</p><h4>Ответ: \(v=200\)</h4>'
                r'<footer>Contact the curator.</footer>')
    assert parser.text("question") == r"Find speed \(v\)."
    assert parser.text("solution") == r"Use \(v=s/t\)."
    assert parser.text("answer") == r"\(v=200\)"
    assert parser.images["question"] == ["diagram.svg"]


def test_native_numbering_keeps_subquestions_together(competition_collector):
    text = (r"\begin{enumerate}[label=1.2.\arabic*]"
            r"\item First question.\begin{enumerate}\item Find speed.\item Find tension.\end{enumerate}"
            r"\item Second question.\end{enumerate}")
    rows = competition_collector.textbook_statements(text)
    assert list(rows) == ["1.2.1", "1.2.2"]
    assert "Find speed." in rows["1.2.1"] and "Find tension." in rows["1.2.1"]
    assert rows["1.2.2"] == "Second question."


def test_competition_subparts_keep_dependencies_and_page_boundaries(monkeypatch):
    import fitz

    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("phy_rl_extract_archives", tools / "phy_rl_extract_archives.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parts = [{"id": "a", "kind": "scalar", "depends_on_parts": []},
             {"id": "b", "kind": "scalar", "depends_on_parts": ["a"]},
             {"id": "c", "kind": "scalar", "depends_on_parts": []},
             {"id": "d", "kind": "qualitative", "depends_on_parts": ["a", "b"]}]
    assert [[p["id"] for p in group] for group in module.grouped_parts(parts)] == [["a", "b"], ["c"]]
    with pytest.raises(ValueError, match="dependency is missing"):
        module.grouped_parts([{"id": "a", "depends_on_parts": ["missing"]}])
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), "Solution of problem 1:")
        page.insert_text((40, 100), "The requested solution.")
        page.insert_text((40, 160), "Problem 2: An excluded neighbouring problem")
        regions = module.page_regions(document, "1", "solution")
        assert len(regions) == 1
        assert "requested solution" in page.get_text("text", clip=regions[0][1])
        assert "excluded neighbouring" not in page.get_text("text", clip=regions[0][1])
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), "An unmarked whole paper")
        with pytest.raises(ValueError, match="no whole-paper fallback"):
            module.page_regions(document, "1", "question")
        assert module.page_regions(document, "1", "question", single_problem=True)
    assert module.transcription_issues({"parts": [{"id": "a", "kind": "scalar",
        "context": "As shown in Fig. 1.", "request": "Find the speed and explain the motion."}]})


def test_competition_batch_selection_and_release_union(monkeypatch, tmp_path):
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("phy_rl_competition_run", tools / "phy_rl_competition_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"source": source, "problem_id": f"{source}-{year}", "year": year,
             "documents": {"question": {"page_count": 1}, "solution": {"page_count": 1}}}
            for source in ["ipho_olimpicos", "apho_archive", "nbpho_olimpicos"] for year in [2023, 2022, 2026]]
    selected = module.select_sources(rows, {"ipho_olimpicos-2023"}, 3)
    assert {r["source"] for r in selected} == {"ipho_olimpicos", "apho_archive", "nbpho_olimpicos"}
    assert all(r["year"] <= 2023 and r["problem_id"] != "ipho_olimpicos-2023" for r in selected)
    first = {"problem_id": "first", "parent_problem_id": "parent", "training_ready": True,
             "checks": {"physics": True}, "source": "ipho_olimpicos", "year": 2023,
             "question": "Find the speed when a bead leaves the circular track.",
             "source_evidence": {"provenance": {"subpart_labels": ["a", "b"]}}}
    second = first | {"problem_id": "second", "question": "Find the tension of a rope supporting a moving mass."}
    path = tmp_path / "accepted.jsonl"
    module.write_rows(path, [first, second])
    accepted, duplicates = module.admitted_union([path])
    assert accepted == [first]
    assert duplicates[0]["problem_id"] == "second"
    hold = {"first": {"question_sha256": module.hashlib.sha256(first["question"].encode()).hexdigest(),
                      "reason": "requested_direction_missing"}}
    module.write_rows(path, [first])
    assert module.admitted_union([path], hold)[0] == []
    with pytest.raises(ValueError, match="no longer matches"):
        module.admitted_union([path], {"first": hold["first"] | {"question_sha256": "stale"}})


def test_russian_archive_uses_end_of_school_year(monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("phy_rl_collect_russian", tools / "phy_rl_collect_russian.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    html = '''<a href="/tasks-phys-9-teor-reg-22-23.pdf">Задания</a>
    <a href="/sol-phys-9-teor-reg-22-23.pdf">Решения</a>
    <a href="/tasks-phys-9-teor-reg-23-24.pdf">Задания</a>
    <a href="/sol-phys-9-teor-reg-23-24.pdf">Решения</a>'''
    papers = module.paper_pairs(html, "https://olimpiada.ru/activity/74/tasks/2022", 9)
    assert len(papers) == 1 and papers[0]["year"] == 2023
    assert papers[0]["stage"] == "reg"
    assert module.paper_pairs(html, "https://olimpiada.ru", 11) == []
    older = '''<a href="/tasks-phys-10-teor-reg-2010-1.pdf">Questions</a>
    <a href="/ans-phys-10-teor-reg-2010-1.pdf">Solutions</a>'''
    assert module.paper_pairs(older, "https://olimpiada.ru", 10)[0]["year"] == 2011
    doc = {"pages": [{"page": 1, "text": "Задача №10-1. Три тигра\n1. Find a speed.\nЗадача №2. Пружина\n"}]}
    headers = module.original_headers({"grade": 10, "paper_id": "russian"}, doc)
    assert [row["problem_number"] for row in headers] == ["10.1", "10.2"]
    doc["pages"].append({"page": 2, "text": "Задача №1. Repeated label\n"})
    with pytest.raises(ValueError, match="Repeated question labels"):
        module.original_headers({"grade": 10, "paper_id": "russian"}, doc)


@pytest.fixture
def bulk_curator(monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("phy_rl_curate", tools / "phy_rl_curate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_release_requires_pinned_revision_and_native_split():
    from physics_rlvr_data.policy import validate_training_policy

    kwargs = dict(source="textbookreasoning_physics", competition="TextbookReasoning physics training material",
                  year=None, split="train", source_split="train",
                  source_revision="ca7ecbec76d01bff2e99f3dc17735b02f87d4e96")
    validate_training_policy(**kwargs)
    with pytest.raises(ValueError, match="source split"):
        validate_training_policy(**(kwargs | {"source_split": "test"}))
    with pytest.raises(ValueError, match="source revision"):
        validate_training_policy(**(kwargs | {"source_revision": "unreviewed"}))


def test_bulk_curation_refuses_stale_question_screen(bulk_curator):
    row = {"problem_id": "example", "source": "textbookreasoning_physics", "source_id": "1",
           "question": "Find the velocity after a constant force acts for one second.", "source_split": "train",
           "reference_answer": "1", "provenance": {"revision": "ca7ecbec76d01bff2e99f3dc17735b02f87d4e96"},
           "benchmark_screening": {"status": "clear", "question_sha256": "different-question"}}
    with pytest.raises(ValueError, match="matching clear source screen"):
        bulk_curator.normalize_source(row)
    coverage = bulk_curator.source_coverage([
        {"source": "ipho_olimpicos"}, {"source": "estonian_physics_olympiad"},
        {"source": "savchenko_solutions"}, {"source": "textbookreasoning_physics"},
    ])
    assert coverage == {"olympiad_count": 2, "olympiad_fraction": 0.5, "ipho_count": 1}
    payload = {"answers": [{"label": "temperature", "value": "22", "unit": "degree Celsius",
                           "answer_type": "numeric", "verifier": "numeric", "equivalent_forms": []},
                          {"label": "length", "value": "r/(4*n-2)", "unit": "m",
                           "answer_type": "symbolic", "verifier": "sympy",
                           "equivalent_forms": ["r / (4 * n - 2)"]}]}
    normalized = bulk_curator.normalize_model_fields(payload)
    assert normalized["answers"][0]["unit"] == "degC"
    assert normalized["answers"][1]["equivalent_forms"] == [normalized["answers"][1]["value"]]
    assert payload["answers"][0]["unit"] == "degree Celsius"


def test_bulk_quality_gate_stops_unknown_charges_and_cost_overrun(bulk_curator, tmp_path):
    args = SimpleNamespace(output=tmp_path, budget=20, pilot_size=50, quality_window_start=0,
                           minimum_model_pass_rate=0.4, minimum_release_rate=0.25,
                           allow_reserved_unconfirmed=False)
    rows = [{"status": "model_checked", "training_ready": True}] * 50
    summary = {"unconfirmed_reserve_usd": 0, "charged_cost_usd": 0.15,
               "projected_cost_all_inputs_usd": 15}
    assert bulk_curator.quality_gate(rows, summary, args) == ""
    assert "Unconfirmed charge" in bulk_curator.quality_gate(rows, summary | {"unconfirmed_reserve_usd": 0.01}, args)
    args.allow_reserved_unconfirmed = True
    assert bulk_curator.quality_gate(rows, summary | {"unconfirmed_reserve_usd": 0.01}, args) == ""
    assert "projects beyond" in bulk_curator.quality_gate(rows, summary | {"projected_cost_all_inputs_usd": 25}, args)
    assert "40%" in bulk_curator.quality_gate([{"status": "review"}] * 50, summary, args)
    args.minimum_model_pass_rate = 0
    assert "final admission" in bulk_curator.quality_gate([{"status": "model_checked"}] * 50, summary, args)
    args.minimum_release_rate = 0
    assert bulk_curator.quality_gate([{"status": "review", "api_error": "StructuredOutputError"}] * 50, summary, args) == ""
    assert "incomplete-response" in bulk_curator.quality_gate(
        [{"status": "review", "api_error": "IncompleteResponseError"}] * 50, summary, args
    )


def test_contract_review_requires_complete_scope_and_source_quotes(bulk_curator):
    question = "Find acceleration and tension. Express tension in terms of mass."
    answers = [{"label": "acceleration"}, {"label": "tension"}]
    review = {"requested_outputs_match": True, "source_reference_supported": True,
              "outputs": [{"label": "acceleration", "request_quote": "Find acceleration and tension."},
                          {"label": "tension", "request_quote": "Express tension in terms of mass."}]}
    assert bulk_curator.contract_checks(question, answers, review)
    assert not bulk_curator.contract_checks(question, answers + [{"label": "tension_numeric"}], review)
    assert not bulk_curator.contract_checks(question, answers, review | {"requested_outputs_match": False})
    assert not bulk_curator.contract_checks(question, answers, review | {"source_reference_supported": False})
    assert not bulk_curator.contract_checks(question, answers, review | {"outputs": review["outputs"][:1]})
    invented = {"label": "tension", "request_quote": "Calculate a numerical tension."}
    assert not bulk_curator.contract_checks(question, answers, review | {"outputs": [review["outputs"][0], invented]})


def test_strong_solver_is_reserved_for_independent_failures(bulk_curator):
    row = {"models": {"auditor": "cheap"}, "self_contained": True, "source_reference_supported": True,
           "checks": {"answer_schema": True, "independent_answer_agreement": False, "required_output_contract": True}}
    assert bulk_curator.needs_blind_resolution(row, "strong")
    assert not bulk_curator.needs_blind_resolution(row, "")
    assert not bulk_curator.needs_blind_resolution(row, "cheap")
    assert not bulk_curator.needs_blind_resolution(row | {"source_reference_supported": False}, "strong")
    assert not bulk_curator.needs_blind_resolution(row | {"checks": row["checks"] | {"answer_schema": False}}, "strong")
    assert not bulk_curator.needs_blind_resolution(row | {"checks": row["checks"] | {"required_output_contract": False}}, "strong")


def test_numeric_bindings_require_source_values_and_global_scope(bulk_curator):
    question = r"Let $\alpha = 1 \times 10^{-6}$. Find the speed."
    entry = {"symbol": "alpha", "value": "1e-6", "source_quote": question.split(". Find")[0] + ".",
             "scope": "entire_task"}
    review = {"question_sha256": bulk_curator.sha(question), "entries": [entry]}
    row = {"question": question, "answers": [{"bindings": {"alpha": "1e-6"}}], "binding_review": review}
    assert bulk_curator.binding_checks(row)
    assert not bulk_curator.binding_checks(row | {"binding_review": {}})
    for change in [{"value": "2e-6"}, {"source_quote": "$alpha=1e-6$"}, {"scope": "initial_condition"}]:
        assert not bulk_curator.binding_checks(row | {"binding_review": review | {"entries": [entry | change]}})
    conflicting = [{"bindings": {"alpha": "2e-6"}}, {"bindings": {"alpha": "1e-6"}}]
    assert not bulk_curator.binding_checks(row | {"answers": conflicting})
