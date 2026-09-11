"""Data for the compression task: JSONL loader plus a seeded synthetic generator.

Record format: {"inputs": {"text": "..."}, "answer": ["Full Name", ...]}
The generator produces passages with 0-3 people, distractor capitalised words (places,
organisations, months) and varied phrasing, so trivial regexes don't score 1.0.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bpto import Dataset

FIRST = ["Charles", "Alice", "Bob", "Priya", "Wei", "Fatima", "Diego", "Olga", "Kwame", "Hana", "Luca", "Amara",
         "Noor", "Tomas", "Ingrid", "Yusuf", "Mei", "Rafael", "Sven", "Zainab"]
LAST = ["Okafor", "Nguyen", "Schmidt", "Rossi", "Haddad", "Kimura", "Petrov", "Mensah", "Larsen", "Castillo",
        "Dubois", "Patel", "Novak", "Farouk", "Byrne", "Tanaka"]
PLACES = ["Nantucket", "Lagos", "Oslo", "Kyoto", "Lima", "Toronto", "Marseille", "Jaipur"]
ORGS = ["Acme Corp", "the Red Cross", "Northwind Traders", "the Ministry of Transport", "Globex"]
MONTHS = ["January", "March", "June", "September", "November"]
TITLES = ["Dr.", "Ms.", "Mr.", "Professor", ""]

TEMPLATES_PERSON = [
    "{t}{name} arrived in {place} in {month}.",
    "The report was filed by {t}{name} on behalf of {org}.",
    "According to {t}{name}, the shipment from {place} was late.",
    "{t}{name} said nothing about {org}.",
    "There once was a traveller from {place}. Their name was {name}.",
    "Everyone expected {t}{name} to speak, but {org} objected.",
]
TEMPLATES_NONE = [
    "The office in {place} closes in {month}.",
    "{org} announced a merger with a rival based in {place}.",
    "Nobody came to the meeting in {place}.",
    "Prices in {place} rose sharply in {month}, {org} reported.",
]


def _name(rng: random.Random, full: bool) -> str:
    f = rng.choice(FIRST)
    return f"{f} {rng.choice(LAST)}" if full else f


def generate_dataset(n: int = 60, seed: int = 0, full_names: bool = True) -> Dataset:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        k = rng.choices([0, 1, 2, 3], weights=[2, 5, 3, 1])[0]
        sents, names = [], []
        for _ in range(k):
            name = _name(rng, full_names)
            names.append(name)
            sents.append(rng.choice(TEMPLATES_PERSON).format(
                t=(rng.choice(TITLES) + " ").lstrip() if rng.random() < 0.4 else "",
                name=name, place=rng.choice(PLACES), org=rng.choice(ORGS), month=rng.choice(MONTHS)))
        for _ in range(rng.randint(0, 2)):
            sents.append(rng.choice(TEMPLATES_NONE).format(
                place=rng.choice(PLACES), org=rng.choice(ORGS), month=rng.choice(MONTHS)))
        rng.shuffle(sents)
        records.append({"id": str(i), "inputs": {"text": " ".join(sents)}, "answer": names})
    return Dataset.from_records(records)


def save_jsonl(ds: Dataset, path: str | Path) -> None:
    with open(path, "w") as f:
        for ex in ds:
            f.write(json.dumps({"id": ex.id, "inputs": ex.inputs, "answer": ex.answer}) + "\n")


def load(path: str | Path) -> Dataset:
    return Dataset.from_jsonl(path)
