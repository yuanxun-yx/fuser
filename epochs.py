import openpyxl
import numpy as np
from pathlib import Path

HEADER = 'Original Social Event'


def read_epochs(xlsx_path: str | Path) -> np.ndarray | None:
    wb = openpyxl.load_workbook(xlsx_path)
    if len(wb.sheetnames) == 1:
        ws = wb.worksheets[0]
    else:
        ws = wb['Processed Events']
    if ws.max_column != 2:
        raise ValueError(f'"{xlsx_path}" has {ws.max_column} columns')
    epochs = None
    for i in range(2, ws.max_row + 1):
        header = ws.cell(row=i, column=1).value
        if not header.startswith(HEADER):
            continue
        values = ws.cell(row=i, column=2).value
        time = [float(s) for s in values.split()]
        time = np.array(time)
        epochs = time.reshape(-1, 2)
        break
    return epochs
