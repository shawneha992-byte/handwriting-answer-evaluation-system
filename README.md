# Handwriting Answer Evaluation System

Upload one question paper, then the handwritten answer sheets of many students. An LLM reads each sheet, the app scores every answer using the marks (and negative marking) printed on the question paper, and you get results out of the exam's total marks.

## How it works

```
Question paper (PDF)  -> LLM reads questions, marks, negative marking -> you confirm
Answer key (JPG/CSV/Excel) -> read into the key                       -> you confirm
Student sheets (PDF, ~50)  -> LLM reads roll no, name, each answer    (one request per sheet)
                           -> scoring.py compares with the key (plain code, repeatable)
                           -> saved in MySQL
Results: table, per-student detail, Excel export.
Sheets the AI is unsure about are NOT guessed: they go to the "Unreadable sheets" report for manual marking.
```

The LLM only reads. Marks are calculated in `backend/app/services/scoring.py`.

## Project layout

```
backend/
  app/main.py                 FastAPI app, also serves the UI
  app/core/config.py          settings from backend/.env
  app/core/database.py        MySQL connection, creates DB and tables on first run
  app/models.py               exams, questions, submissions, answers
  app/routes/exams.py         upload, results, export, manual marks
  app/routes/system.py        dashboard, config, health
  app/services/llm_service.py Gemini calls (and a mock mode)
  app/services/scoring.py     marks and negative marking
  app/services/processing.py  background pipeline per sheet
  app/services/export.py      Excel workbook
  tests/test_pipeline.py      end-to-end test
frontend/index.html           the UI (calls the API)
```

## Run it

1. Create a MySQL server (or use the XAMPP/MySQL you already have). The database is created automatically.
2. Set up Python:
   ```
   cd backend
   python -m venv venv
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   copy .env.example .env         (Windows)   |   cp .env.example .env
   ```
3. Edit `backend/.env`: set your MySQL user and password in `DATABASE_URL`, and your free Gemini key in `GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey).
4. Start it: `uvicorn app.main:app --reload`
5. Open http://127.0.0.1:8000

**Try it without a key:** set `LLM_PROVIDER=mock` in `.env`. The app then uses fake data, so upload any dummy PDFs to see the whole flow.

## Tests

```
pip install pytest httpx
pytest -q
```

## Notes

- Free Gemini limits change, so check https://ai.google.dev/gemini-api/docs/rate-limits. The app waits `GEMINI_DELAY_SECONDS` between sheets and retries on rate-limit errors.
- Free-tier prompts may be used by Google to improve its products. Use dummy or consented sheets while developing.
- A student's score never goes below zero. Unreadable answers score zero and the sheet is listed for review.
- Not built yet: descriptive (long) answers, user login, PDF report export.
