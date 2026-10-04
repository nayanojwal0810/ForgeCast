"""Deterministic telemetry replay engine simulating sequential industrial telemetry."""

import csv
from collections.abc import Iterator
from pathlib import Path
import time
from typing import Union

from forgecast.ingestion.contract import TelemetryRecord
from forgecast.ingestion.validator import TemporalValidator


class TelemetryReplay:
    """Historical telemetry replay engine.

    Streams telemetry records from the raw CSV source in strict chronological order,
    validates every record and temporal continuity, and yields normalized TelemetryRecord events.
    """

    def __init__(
        self,
        csv_path: Union[str, Path] = "data/raw/Steel_industry_data.csv",
        delay_seconds: float = 0.0,
        limit: int | None = None,
        validator: TemporalValidator | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if delay_seconds < 0.0:
            raise ValueError(f"delay_seconds cannot be negative, got {delay_seconds}")
        self.delay_seconds = float(delay_seconds)
        self.limit = limit
        self.validator = validator or TemporalValidator()
        self.emitted_count: int = 0

    def run(self) -> Iterator[TelemetryRecord]:
        """Run sequential replay, yielding validated TelemetryRecord events one at a time.

        Yields:
            TelemetryRecord: Chronologically ordered, validated observation.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            TelemetryValidationError: If any row fails contract or temporal continuity.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Source telemetry dataset not found at '{self.csv_path}'")

        self.validator.reset()
        self.emitted_count = 0

        with open(self.csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_idx, raw_row in enumerate(reader):
                if self.limit is not None and self.emitted_count >= self.limit:
                    break

                if self.delay_seconds > 0.0 and self.emitted_count > 0:
                    time.sleep(self.delay_seconds)

                record = self.validator.validate_record(raw_row, record_index=row_idx)
                self.emitted_count += 1
                yield record

    def __iter__(self) -> Iterator[TelemetryRecord]:
        """Support standard iteration protocol."""
        return self.run()
