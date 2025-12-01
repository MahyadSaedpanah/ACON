import csv
import os
from typing import Optional, Dict, Any


class ContrastiveLogger:
    def __init__(self, log_path: str = "contrastive_logs.csv"):
        self.log_path = log_path
        self._file_exists = os.path.exists(self.log_path)
        self._header_written = self._file_exists

    def log(self, row: Dict[str, Any]):
        clean_row = {}
        for k, v in row.items():
            try:
                if hasattr(v, "item") and callable(v.item):
                    v = v.item()
            except Exception:
                pass
            clean_row[k] = v

        write_header = False
        if not self._header_written:
            write_header = True
            self._header_written = True

        # append mode
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(clean_row)
