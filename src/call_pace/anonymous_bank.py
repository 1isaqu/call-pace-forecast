"""The public "Anonymous Bank" call-center data: fetch, clean and shape it for this package.

The data is a call-by-call record of the telephone call center of an Israeli retail bank for the
whole of 1999: 12 monthly files, about 444,000 records. It was organized by Ilan Guedj and Avi
Mandelbaum and is published by the Technion SEE Laboratory
(https://seelab.net.technion.ac.il/data/).
The authors state that the data is free for use and ask users to notify Avi Mandelbaum and to
acknowledge the source. There is no formal licence. This repository does not contain the data.

Nothing here runs unless you call it: the library opens no connection by itself, and the tests use
small inline fixtures. `fetch` needs the optional extra `realdata`, because the archive is a RAR
file.

Events are call arrivals, timed at the moment a call enters the voice response unit, with one bucket
per service type. The center is staffed Sunday to Thursday from 7:00 to midnight.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .errors import ContractError, DataSourceError

ARCHIVE_URL = "https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.rar"
DOCUMENTATION_URL = (
    "https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.pdf"
)
ARCHIVE_SHA256 = "110527823a296e3163566f5d10d687e13859c60d963fd403d4623d89c7cadc4b"
DOCUMENTATION_SHA256 = "77816db694fdee313a7d612e22199d87632ddca5a82f17c3258bd6799ff6e045"
MONTH_SHA256 = {
    "January1999.txt": "0274cfb02703a5ffadc7d381064cb177ac0b4be2fe6ce439283e80d95de9103f",
    "February1999.txt": "4004237dcb32355816e9bf98b9a61401ec4ded56c5c66c4a51d3e78f9802cf46",
    "March1999.txt": "3d95552ceff76390d8bd78aa7d166fb2fd077cb74a8b08f2000ab1334ce9a118",
    "April1999.txt": "cf51f7f2c8d19ae68181d0b51f3e41ce58a5d5d97c4d81dada728f061fd6211d",
    "May1999.txt": "9efbd488959bbd8cf3489140fa599784364d41f334bc9a1db29e95b611308325",
    "June1999.txt": "29c6c4f2202c2a1771ed0c490ba708cf3fc798afe664fd3058d78df76adec495",
    "July1999.txt": "80012ee43912072e26c3fc00369a925e75102b61776e25130bfac8750addd9b0",
    "August1999.txt": "2a61cf61d52cc62d18d0fe96f89aae8b5378d5b345775ecfcbe704bf30d9af60",
    "September1999.txt": "a6d8a914d232b02b171279033ec8c87c3f607ed218a86c36c06847769a15f00d",
    "October1999.txt": "078b3faccfbc86b8ca97d8c0ad2750119e6a36fd5184d058330e16f698a81932",
    "November1999.txt": "20df9fc55b894fd8695540e5060e0ed74dc0c6c7ad596d4a97113932e2483511",
    "December1999.txt": "1c723ffc05d1567d59025b1b68079739bc4c339f80aafba4398ee80f639a3209",
}

# The six service types of the authors' documentation.
SERVICE_TYPES = {
    "PS": "regular activity",
    "PE": "regular activity in English",
    "IN": "internet help desk",
    "NE": "stock exchange activity",
    "NW": "prospective customers asking for information",
    "TT": "customers who asked to be called back and were called by the system",
}
_OUTCOMES = ("AGENT", "HANG", "PHANTOM")
_DATE = r"\d{6}"
_TIME = r"\d{1,2}:\d{2}:\d{2}"

# Python weekday numbers (Monday = 0): Sunday to Thursday.
OPERATING_WEEKDAYS = (0, 1, 2, 3, 6)


@dataclass(frozen=True)
class DataQuality:
    """What `clean_calls` kept and dropped, so every exclusion can be reported."""

    raw_rows: int
    malformed_rows: int
    type_trailing_space: int
    phantom_rows: int
    undocumented_type_rows: int
    undocumented_types: tuple[str, ...]
    calls_kept: int
    calendar_days: int


def anonymous_bank_config(**changes: object) -> Config:
    """The center's calendar and the walk-forward protocol of docs/real-data-evaluation.md."""
    settings: dict[str, object] = {
        "start_hour": 7,
        "end_hour": 23,
        "operating_weekdays": OPERATING_WEEKDAYS,
        "first_window_hours": 3,
        "n_folds": 6,
        "fold_size_days": 20,
        "min_train_days": 100,
    }
    settings.update(changes)
    return Config(**settings)


def clean_calls(raw: pd.DataFrame) -> tuple[pd.DataFrame, DataQuality]:
    """Drop what the documentation or the data rules out, and count every drop.

    `raw` holds the monthly files as read with every column as text. A row is malformed when its
    date, arrival time or outcome has the wrong shape: in the original files a few rows lack the
    `type` cell, so the cells after it shift left. `PHANTOM` calls are to be ignored according to
    the documentation. A type the documentation does not describe is dropped. A type with a
    trailing space is fixed. The result has the columns `type`, `date` (Timestamp) and `hour`.
    """
    needed = {"type", "date", "vru_entry", "outcome"}
    missing = needed - set(raw.columns)
    if missing:
        raise ContractError(f"call records lack the columns {sorted(missing)}")
    well_formed = (
        raw["date"].str.fullmatch(_DATE)
        & raw["vru_entry"].str.fullmatch(_TIME)
        & raw["outcome"].isin(_OUTCOMES)
    ).fillna(False)
    calls = raw[well_formed].copy()
    trailing = int((calls["type"] != calls["type"].str.strip()).sum())
    calls["type"] = calls["type"].str.strip()
    phantom = int((calls["outcome"] == "PHANTOM").sum())
    calls = calls[calls["outcome"] != "PHANTOM"]
    unknown = ~calls["type"].isin(SERVICE_TYPES)
    undocumented = tuple(sorted(calls.loc[unknown, "type"].unique()))
    undocumented_rows = int(unknown.sum())
    calls = calls[~unknown].copy()
    calls["date"] = pd.to_datetime(calls["date"], format="%y%m%d")
    calls["hour"] = calls["vru_entry"].str.split(":").str[0].astype(int)
    quality = DataQuality(
        raw_rows=len(raw),
        malformed_rows=int((~well_formed).sum()),
        type_trailing_space=trailing,
        phantom_rows=phantom,
        undocumented_type_rows=undocumented_rows,
        undocumented_types=undocumented,
        calls_kept=len(calls),
        calendar_days=int(calls["date"].nunique()),
    )
    return calls[["type", "date", "hour"]].reset_index(drop=True), quality


def hourly_counts(calls: pd.DataFrame) -> pd.DataFrame:
    """Arrivals per (bucket, date, hour), in the format of docs/data-contract.md."""
    counts = calls.groupby(["type", "date", "hour"]).size().rename("count").reset_index()
    return counts.rename(columns={"type": "bucket"})


def read_calls(folder: str | Path) -> tuple[pd.DataFrame, DataQuality]:
    """Read the 12 monthly files that `fetch` wrote into `folder` and clean them."""
    folder = Path(folder)
    absent = [name for name in MONTH_SHA256 if not (folder / name).is_file()]
    if absent:
        raise DataSourceError(f"missing files in {folder}: {absent}; run fetch first")
    frames = [
        pd.read_csv(folder / name, sep="\t", dtype=str).rename(columns={"vru+line": "vru_line"})
        for name in MONTH_SHA256
    ]
    return clean_calls(pd.concat(frames, ignore_index=True))


def check_hash(data: bytes, expected: str, what: str) -> None:
    """Raise `DataSourceError` unless the SHA-256 of `data` equals `expected`."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DataSourceError(f"{what}: SHA-256 is {actual}, expected {expected}")


def extract_archive(archive: Path, folder: Path) -> list[Path]:
    """Write the monthly files of the RAR archive into `folder` after checking each hash."""
    try:
        from unrar.cffi import rarfile
    except ImportError as exc:
        raise DataSourceError(
            'reading the RAR archive needs the extra: pip install -e ".[realdata]"'
        ) from exc
    rar = rarfile.RarFile(str(archive))
    contents = {name: rar.read(name) for name in MONTH_SHA256}
    for name, expected in MONTH_SHA256.items():
        check_hash(contents[name], expected, name)  # nothing is written unless every file matches
    written = []
    for name, content in contents.items():
        (folder / name).write_bytes(content)
        written.append(folder / name)
    return written


def fetch(
    folder: str | Path,
    urlopen: Callable[..., object] = urllib.request.urlopen,
    timeout: float = 120.0,
) -> Path:
    """Download the archive and the documentation from the Technion host and extract the data.

    Every download is checked against the pinned SHA-256. Files that are already there and match
    are not downloaded again. The data must stay out of version control (`data/` is ignored).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for url, expected, name in (
        (ARCHIVE_URL, ARCHIVE_SHA256, "AnonymousBank.rar"),
        (DOCUMENTATION_URL, DOCUMENTATION_SHA256, "AnonymousBank.pdf"),
    ):
        target = folder / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        try:
            with urlopen(url, timeout=timeout) as response:  # noqa: S310 (fixed https URLs)
                data = response.read()
        except OSError as exc:
            raise DataSourceError(f"could not download {url}: {exc}") from exc
        check_hash(data, expected, name)
        target.write_bytes(data)
    extract_archive(folder / "AnonymousBank.rar", folder)
    return folder
