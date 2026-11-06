from app.services.answer_key_reader import extract_answers_from_text


def test_extracts_final_answer_lines_from_solved_worksheet():
    text = """Q1. 2,345 + 1,256
2,345 + 1,256 = 3,601
Answer = 3,601
Q2. 4,678 + 2,145
Answer = 6,823
"""
    assert extract_answers_from_text(text) == {"1": "3601", "2": "6823"}


def test_extracts_numbered_answer_key():
    text = """1. 3601
2) 6823
Q3: 16245
"""
    assert extract_answers_from_text(text) == {"1": "3601", "2": "6823", "3": "16245"}


def test_does_not_treat_worked_equation_as_final_answer():
    text = """Q1. 10 + 20
10 + 20 = 30
"""
    assert extract_answers_from_text(text) == {}
