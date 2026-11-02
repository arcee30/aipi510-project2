# AI and Human Writing Perception Survey

This project studies how respondents rate writing quality and whether they
perceive six passages as AI-generated or human-written.
Responses were collected through a survey form; no custom data collection
script was used.

## Data

- `data/raw/survey_raw.xlsx` is the original survey export. It contains 32
  responses and their timestamps. The raw file is kept unchanged.
- `data/clean/survey_responses.csv` has one row per response and a numeric
  `response_id`. It preserves the 14 survey answers and omits timestamps.
- `data/clean/passage_answer_key.csv` records the study team's ground truth:
  passages 1, 4, and 6 are human-written; passages 2, 3, and 5 are AI-generated.

For each passage, `pN_quality` is rated from 1 (awful) to 5 (excellent).
`pN_origin_rating` is rated from 1 (certainly AI) to 10 (certainly human).
`native_english` is the respondent's Yes/No answer. `english_proficiency`
preserves the respondent's text answer; 16 responses leave it blank.

## Reproduce the cleaned data

Use Python 3.10 or later from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/clean_survey.py
```

The script reads `data/raw/survey_raw.xlsx` and regenerates both files in
`data/clean/`. It assigns response IDs in source row order, validates the
rating ranges, and does not modify the raw workbook. The answer key is a
study-team input recorded in the script, not inferred from responses.

Analysis scripts and results can be added in separate pull requests.
