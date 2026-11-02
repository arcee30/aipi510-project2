"""Build analysis-ready CSV files from the original survey workbook."""

import csv
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "survey_raw.xlsx"
CLEAN_DIR = ROOT / "data" / "clean"

# Ground truth supplied by the study team, in passage order.
CORRECT_ORIGINS = ("human", "ai", "ai", "human", "ai", "human")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    sheet = load_workbook(RAW_FILE, read_only=True, data_only=True).active
    if sheet.max_column != 15:
        raise ValueError(f"Expected 15 source columns, found {sheet.max_column}")

    headers = ["response_id", "native_english", "english_proficiency"]
    for passage in range(1, 7):
        headers += [f"p{passage}_quality", f"p{passage}_origin_rating"]

    responses: list[list[object]] = []
    for source_row in sheet.iter_rows(min_row=2, values_only=True):
        if all(value is None for value in source_row):
            continue

        # The raw timestamp is excluded because it is not needed for this analysis.
        answers = list(source_row[1:15])
        if len(answers) != 14:
            raise ValueError("A survey row has the wrong number of answers")
        for passage in range(6):
            quality, origin = answers[2 + 2 * passage : 4 + 2 * passage]
            if not isinstance(quality, int) or not 1 <= quality <= 5:
                raise ValueError(f"Invalid quality score in response {len(responses) + 1}")
            if not isinstance(origin, int) or not 1 <= origin <= 10:
                raise ValueError(f"Invalid origin score in response {len(responses) + 1}")
        responses.append([len(responses) + 1, *answers])

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(CLEAN_DIR / "survey_responses.csv", headers, responses)
    write_csv(
        CLEAN_DIR / "passage_answer_key.csv",
        ["passage_id", "correct_origin"],
        [[passage, origin] for passage, origin in enumerate(CORRECT_ORIGINS, start=1)],
    )
    print(f"Created cleaned data for {len(responses)} responses and 6 passages")


if __name__ == "__main__":
    main()
