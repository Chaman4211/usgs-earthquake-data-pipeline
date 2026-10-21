"""Simulate streaming by polling a USGS earthquake feed on a timer."""

import argparse
import time
from pathlib import Path

import pandas as pd

from quakes.fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return rows with unseen (id, updated) pairs and update seen in place."""
    new_indices = []

    for index, row in df.iterrows():
        pair = (row["id"], row["updated"])

        if pair not in seen:
            new_indices.append(index)
            seen.add(pair)

    return df.loc[new_indices].copy()


def run_stream(
    feed: str,
    interval_s: int,
    duration_s: int,
    out_path: str,
) -> None:
    """Poll a feed, append unseen event versions as JSONL, and report progress."""
    if interval_s <= 0:
        raise ValueError("interval_s must be greater than zero.")
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative.")

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    start = time.monotonic()

    while time.monotonic() - start < duration_s:
        poll_start = time.monotonic()

        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)
            new_df = filter_unseen(df, seen)

            if not new_df.empty:
                with output.open("a", encoding="utf-8") as file:
                    new_df.to_json(
                        file,
                        orient="records",
                        lines=True,
                        date_format="iso",
                    )

            timestamp = time.strftime("%H:%M:%S")
            print(
                f"[{timestamp}] {len(new_df)} new / {len(df)} fetched",
                flush=True,
            )

        except Exception as exc:
            # A failed poll should not terminate the stream.
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Poll failed: {exc}", flush=True)

        elapsed = time.monotonic() - start
        remaining = duration_s - elapsed

        if remaining <= 0:
            break

        poll_elapsed = time.monotonic() - poll_start
        time.sleep(min(max(0, interval_s - poll_elapsed), remaining))


def main() -> None:
    """Parse command-line arguments and start the stream."""
    parser = argparse.ArgumentParser(
        description="Poll a USGS earthquake feed and save new event versions."
    )
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument("--out", default="data/stream/stream.jsonl")

    args = parser.parse_args()

    run_stream(
        feed=args.feed,
        interval_s=args.interval,
        duration_s=args.duration,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
