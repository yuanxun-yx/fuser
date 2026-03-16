import openpyxl
import numpy as np
from pathlib import Path


def read_epochs(xlsx_path: str | Path) -> dict[str, np.ndarray]:
    def condition_tag(header: str) -> str:
        words = [w.lower() for w in header.split()]
        if words[0] != 'modified':
            raise ValueError(f'"{header}" is not valid')
        return words[1]

    wb = openpyxl.load_workbook(xlsx_path)
    if len(wb.sheetnames) == 1:
        ws = wb.worksheets[0]
    else:
        ws = wb['Processed Events']
    if ws.max_column != 2:
        raise ValueError(f'"{xlsx_path}" has {ws.max_column} columns')
    epochs_by_condition = {}
    for i in range(2, ws.max_row + 1):
        event_type = ws.cell(row=i, column=1).value
        try:
            condition = condition_tag(event_type)
        except ValueError:
            continue
        if condition in epochs_by_condition:
            raise ValueError(f'more than one {condition} found in "{xlsx_path}"')
        values = ws.cell(row=i, column=2).value
        time = [int(s) for s in values.split()]
        time = np.array(time)
        epochs_by_condition[condition] = time.reshape(-1, 2)
    return epochs_by_condition
