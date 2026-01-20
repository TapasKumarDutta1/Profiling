import re
from typing import Optional

import numpy as np
import pandas as pd


def _pick_subtitle(subtitles):
    if not isinstance(subtitles, list):
        return None
    if len(subtitles) == 1:
        return subtitles[0].get("name")
    if subtitles[0].get("name", "").startswith("Including topics"):
        return subtitles[1].get("name")
    return subtitles[0].get("name")


def _details_label(details):
    if isinstance(details, list) and details:
        first = details[0]
        if isinstance(first, dict):
            return first.get("name")
    return None


def _extract_url(title):
    if not isinstance(title, str):
        return None
    match = re.search(r"https?://\S+", title)
    return match.group(0).rstrip('"') if match else None


def _combine_row(row):
    parts = []
    if pd.notna(row.get("title")):
        parts.append(row["title"])
    if pd.notna(row.get("time_of_day")):
        parts.append(f"during {row['time_of_day']}")
    if pd.notna(row.get("source")):
        parts.append(f"from {row['source']}")
    return " ".join(parts)


def clean_search_history_json(
    input_path,
    output_csv: Optional[str] = "cleaned_search_history.csv",
    days: Optional[int] = 30,
):
    """
    Load a Google search history JSON export and write a cleaned CSV.

    Returns the cleaned dataframe.
    """
    df = pd.read_json(input_path, convert_dates=["time"])

    for col in ["products", "activityControls"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: ", ".join(val) if isinstance(val, list) else val
            )

    if "locationInfos" in df.columns:
        locations = (
            df[["time", "title", "locationInfos"]]
            .explode("locationInfos")
            .reset_index(drop=True)
        )
        location_details = pd.json_normalize(locations["locationInfos"])
        df = pd.concat([df.drop(columns="locationInfos"), location_details], axis=1)

    if "subtitles" in df.columns:
        df["subtitle_text"] = df["subtitles"].apply(_pick_subtitle)
        df = df.drop(columns=["subtitles"])

    if "details" in df.columns:
        df["details_label"] = df["details"].apply(_details_label)
        df = df.drop(columns=["details"])

    drop_cols = [
        col
        for col in ["header", "products", "activityControls", "name", "sourceUrl", "url"]
        if col in df.columns
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time", ascending=False)

    if "title" in df.columns:
        df["extracted_url"] = df["title"].apply(_extract_url)
        df["title"] = (
            df["title"]
            .str.replace(r"https?://\S+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    df["date"] = df["time"].dt.day
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    hour = df["time"].dt.hour
    df["time_of_day"] = np.select(
        [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 17)],
        ["morning", "noon"],
        default="night",
    )

    df["combined"] = df.apply(_combine_row, axis=1)
    df = df[["combined", "title", "time"]]
    df = df.sort_values("time", ascending=False)

    if days is not None:
        cutoff = df["time"].max() - pd.Timedelta(days=days)
        df = df[df["time"] >= cutoff]

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df
