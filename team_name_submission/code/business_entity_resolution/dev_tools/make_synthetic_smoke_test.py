#!/usr/bin/env python3
"""
Build a small, SYNTHETIC, self-consistent train set to smoke-test the
pipeline end to end (train.py -> predict.py -> evaluate.py -> validator).

Why this exists: the sample rows the challenge provided for
train_source1/2/3.tsv and train_ground_truth.tsv are independent
head-of-file excerpts of much larger files - they do not correspond to the
same entities, so training against them directly yields zero true matches
(you can check: none of the sample source1 entity_ids appear in the sample
ground_truth file). That is expected for a size-limited preview and not a
bug in this pipeline.

This script instead fabricates a small, internally-consistent dataset -
a set of "true" businesses, each rendered three times with source-specific
noise (abbreviation swaps, casing, punctuation, word order, dropped
address components) - purely so the training/evaluation code path can be
exercised and sanity-checked before pointing it at the real, full-size
challenge data. It is NOT part of the submitted model or a claim about
real-world accuracy.
"""

from __future__ import annotations

import csv
import os
import random

random.seed(7)

# (canonical name, canonical address, country)
SEED_BUSINESSES = [
    ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US"),
    ("Prime Money Corporation", "17560 Ellis Road, Tahlequah, OK", "US"),
    ("B+ Retail Incorporated", "1712 Montebello Avenue, Phoenix, AZ", "US"),
    ("Custom Wealth Services LLC", "5559 Orville Avenue, Columbus, OH", "US"),
    ("Nexus Anchor Rain Inc", "1111 Church Street, Unit 2007, Nashville, TN", "US"),
    ("Moore Bitwise Inc", "337 Oakland Avenue, Michigan City, IN", "US"),
    ("Crystal Lending PC", "11643 Prosperity Road, South Jordan, UT", "US"),
    ("Kelly Advisory Inc", "301 1st Street, Chokio, MN", "US"),
    ("Helios Health", "66 Edgewood Street, Bridgeport, CT", "US"),
    ("Health Fellowship Partners", "601 Oleander Circle, Virginia Beach City, VA", "US"),
    ("Callicoat and Dailey Inc", "833 Reliance Street, Charlotte, NC", "US"),
    ("Scott Eagle Inc", "4828 Hedges Avenue, Kansas City, MO", "US"),
    ("Schaefer Michael and Silver Associates", "32876 Circle Drive, Millsboro, DE", "US"),
    ("Colline Molitor Sea LLC", "5620 Trinity Road, Unit 103, Raleigh, NC", "US"),
    ("Global Hovnanian LLC", "4024 39th Avenue, Seattle, WA", "US"),
    ("Pacific Learning Laboratories LLC", "4810 Nassau Avenue, Sand Springs, OK", "US"),
    ("Cascade Allied Telecom LLC", "165 Linden Street, Unit 106, Wellesley, MA", "US"),
    ("Apex Incorporated", "108 Richardson Street, Bethany, WV", "US"),
    ("Foot and Ankle Allied Center LLC", "1216 Preston Avenue, Charlottesville City, VA", "US"),
    ("Joeann Hills Park Inc", "10101 Threave Road, Unit 301, Raleigh, NC", "US"),
    ("Primary Care Physicians Inc", "53 Park Lane, Wellsville, NY", "US"),
    ("Global Ministries", "74 429, Stockdale, TX", "US"),
    ("Siobhan's Bike Shop LLC", "1644 Crownsville Road, Fl 0, Crownsville, MD", "US"),
    ("Corner Hypnosis", "26 Stone Street, Unit 3, Beverly, MA", "US"),
    ("Bautista Properties LLC", "7196 State Route 14, Mcleansboro, IL", "US"),
    ("Prabhav Business Center", "797 Lake Town Block A, Kolkata, West Bengal", "India"),
    ("Smart Healthcare Private Limited", "303 3rd Floor Sakar 5, Ashram Road, Ahmedabad, Gujarat", "India"),
    ("Iris Advisors", "House No 777 Raghubir Bhawan, Sarita Vihar, Delhi", "India"),
    ("Green Logistics Private Limited", "E-7 Second Floor, New Delhi, Delhi", "India"),
    ("Angad Multitrade Ltd", "Building 37/4 First Floor, Vazhakkala, Ernakulam, Kerala", "India"),
    ("Indian Brothers Private Limited", "17/1 Lower Ground Floor, Lalbagh Road, Bangalore, Karnataka", "India"),
    ("Dream Construction Limited", "16-11-23/37/A 2nd Floor, Hyderabad, Telangana", "India"),
    ("Shiva Management Private Limited", "20-6-3/12 Dantuluri Vari Street, Vijayawada, Andhra Pradesh", "India"),
    ("Orchid Renewable Ltd", "House No 26-352 Kadavakollu Bhavanam, Machilipatanam, Krishna, Andhra Pradesh", "India"),
    ("Blue Steels Limited", "Floor 2 Plot 750, Satnam Nagar, Mumbai, Maharashtra", "India"),
    ("North Storage Care", "2804 Tower A Sarova, Samata Nagar, Mumbai, Maharashtra", "India"),
    ("Secunderabad Park Private Limited", "Plot 53 Flat 402, Sr Estate Rainbow Colony, Secunderabad, Telangana", "India"),
    ("Rajani Enterprises of Kannur", "46/1106 Dhanalakshmi Road, Kannur, Kerala", "India"),
    ("Mayur Hospital of Gondal", "C/O Dwarkadhis Enterprise, Gondal, Rajkot, Gujarat", "India"),
    ("Team Ecole Association", "175 Boulevard du President Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France"),
    ("ZNB Club SARL", "5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine", "France"),
    ("Thermal et Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France", "France"),
    ("Grain et Fils", "329 Avenue de Dunkerque, Lille, Hauts-de-France", "France"),
    ("Elephant Centre EURL", "30 Rue Lachassaigne, Bordeaux, Nouvelle-Aquitaine", "France"),
]

# Extra pure-noise "distractor" businesses that appear ONLY in S2/S3 (no S1
# match at all) so the model also has to learn to say "no match".
DISTRACTORS = [
    ("Random Distinct Ventures LLC", "42 Nowhere Lane, Boise, ID", "US"),
    ("Totally Unrelated Traders Pvt Ltd", "Sector 9, Gurugram, Haryana", "India"),
    ("Une Autre Societe SARL", "12 Rue Inconnue, Nantes, Pays de la Loire", "France"),
    ("Distinct Pharmacy Group", "900 Different Ave, Reno, NV", "US"),
    ("Alpha Nonmatch Enterprises", "77 Other Street, Tulsa, OK", "US"),
]

SUFFIX_SWAPS = {
    "incorporated": "inc", "inc": "incorporated", "llc": "l.l.c.",
    "limited": "ltd", "ltd": "limited", "private limited": "pvt ltd",
    "corporation": "corp",
}


def noisy_name(name: str, level: int) -> str:
    n = name
    low = n.lower()
    for a, b in SUFFIX_SWAPS.items():
        if a in low:
            idx = low.find(a)
            n = n[:idx] + b + n[idx + len(a):]
            break
    if level >= 2:
        n = n.replace(" and ", " & ")
        words = n.split()
        if len(words) > 3 and random.random() < 0.5:
            # swap two interior words to simulate reordering
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
            n = " ".join(words)
    if level >= 3 and len(n) > 6 and random.random() < 0.4:
        pos = random.randint(2, len(n) - 2)
        n = n[:pos] + n[pos + 1:]  # drop one character (typo)
    return n


def noisy_address(addr: str, level: int) -> str:
    a = addr
    a = a.replace("Road", "Rd").replace("Street", "St").replace("Avenue", "Ave")
    a = a.replace("Drive", "Dr").replace("Boulevard", "Blvd")
    if level >= 2:
        parts = [p.strip() for p in a.split(",")]
        random.shuffle(parts)
        a = ", ".join(parts)
    if level >= 3 and random.random() < 0.5:
        parts = [p.strip() for p in a.split(",")]
        if len(parts) > 1:
            parts = parts[:-1]  # drop the last component (e.g. state)
        a = ", ".join(parts)
    return a.upper() if level == 3 else a


def build(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []

    for i, (name, addr, country) in enumerate(SEED_BUSINESSES):
        s1_id = f"S1-{100000 + i}"
        s1_rows.append((s1_id, name, addr, country))

        matched = []
        # ~85% of entities get an S2 match, ~75% get an S3 match, a few
        # entities are deliberate singletons (no S2/S3 record at all).
        if random.random() < 0.85:
            s2_id = f"S2-{200000 + i}"
            s2_rows.append(
                (s2_id, noisy_name(name, 2), noisy_address(addr, 1), country)
            )
            matched.append(s2_id)
        if random.random() < 0.75:
            s3_id = f"S3-{300000 + i}"
            s3_rows.append(
                (s3_id, noisy_name(name, 3), noisy_address(addr, 3), country)
            )
            matched.append(s3_id)
        # occasionally a *second* S3 duplicate of the same business exists
        if random.random() < 0.15:
            s3_id2 = f"S3-{300500 + i}"
            s3_rows.append(
                (s3_id2, noisy_name(name, 2), noisy_address(addr, 2), country)
            )
            matched.append(s3_id2)

        gt_rows.append((s1_id, ",".join(matched)))

    for j, (name, addr, country) in enumerate(DISTRACTORS):
        s2_rows.append((f"S2-{900000 + j}", name, addr, country))
        s3_rows.append((f"S3-{900000 + j}", noisy_name(name, 2), addr, country))

    def dump(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            for r in rows:
                w.writerow(r)

    dump(os.path.join(out_dir, "train_source1.tsv"),
         ["entity_id", "business_name", "business_address", "country"], s1_rows)
    dump(os.path.join(out_dir, "train_source2.tsv"),
         ["entity_id", "business_name", "business_address", "country"], s2_rows)
    dump(os.path.join(out_dir, "train_source3.tsv"),
         ["entity_id", "business_name", "business_address", "country"], s3_rows)
    dump(os.path.join(out_dir, "train_ground_truth.tsv"),
         ["source1_entity_id", "matched_entity_ids"], gt_rows)

    print(f"Synthetic smoke-test set written to {out_dir}")
    print(f"  S1={len(s1_rows)} S2={len(s2_rows)} S3={len(s3_rows)} "
          f"labeled_S1={len(gt_rows)}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "dataset/dev_synthetic/train"
    build(out)
