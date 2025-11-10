#!/usr/bin/env python
"""Prepare MusicGen fine-tuning dataset from raw beat segments and metadata.

Usage example:
  python prepare_finetune_dataset.py \
      --raw-metadata ../data/raw-data/raw-metadata.json \
      --audio-dir ../data/raw-data/beat-segment \
      --output-dir ../data \
      --train-ratio 0.8

This script will:
  * Load the raw metadata (style/tag information per beat)
  * Map each metadata entry to its corresponding split audio segments
  * Resample audio to the target sampling rate and ensure mono channel
  * Split outputs into train/valid folders
  * Generate metadata.json compatible with the LoRA training pipeline
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


@dataclass
class Sample:
    """Dataclass to hold information about each processed sample."""

    source_path: Path
    text: str
    base_name: str
    segment_idx: int


def build_prompt(style: str, tag: str) -> str:
    """Compose the text prompt from style and tag fields.

    The format follows: "{style} hiphop beat, {tag}".
    Both style and tag are lower-cased and stripped, commas in style are replaced with spaces.
    """

    def clean_text(value: str) -> str:
        cleaned = value.strip().lower()
        # remove redundant quotes and double spaces
        cleaned = cleaned.replace("'", "").replace(",", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned

    style_text = clean_text(style)
    tag_text = clean_text(tag)

    parts = []
    if style_text:
        parts.append(f"{style_text} hiphop beat")
    else:
        parts.append("hiphop beat")
    if tag_text:
        parts.append(tag_text)

    return ", ".join(parts)


def collect_samples(raw_metadata: Path, audio_dir: Path) -> List[Sample]:
    """Collect samples by pairing metadata with audio segments."""

    with raw_metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples: List[Sample] = []
    for entry in metadata:
        base = entry.get("audio")
        style = entry.get("style", "")
        tag = entry.get("tag", "")

        if not base:
            continue

        prompt = build_prompt(style, tag)

        for segment_idx in (1, 2):
            source_path = audio_dir / f"{base}-{segment_idx}.wav"
            if not source_path.exists():
                raise FileNotFoundError(f"Missing audio file: {source_path}")
            samples.append(Sample(source_path=source_path, text=prompt, base_name=base, segment_idx=segment_idx))

    if not samples:
        raise ValueError("No samples collected. Check metadata or audio directory.")

    return samples


def process_audio(sample: Sample, target_sr: int) -> np.ndarray:
    """Load, resample, and convert the audio to mono float32."""

    audio, sr = librosa.load(sample.source_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def save_audio(audio: np.ndarray, sample_rate: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate)


def prepare_dataset(
    raw_metadata: Path,
    audio_dir: Path,
    output_dir: Path,
    train_ratio: float,
    sample_rate: int,
    seed: int,
    prefix: str,
) -> None:
    """Main preparation pipeline."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Clean up existing output files
    for subset_name in ["train", "valid"]:
        output_subset_dir = output_dir / subset_name
        for file_path in output_subset_dir.glob("*.wav"):
            file_path.unlink()
        metadata_path = output_subset_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

    samples = collect_samples(raw_metadata, audio_dir)
    random.seed(seed)
    # Group by base_name (original beat id) so that both segments from the same beat stay together
    groups = {}
    for s in samples:
        groups.setdefault(s.base_name, []).append(s)
    unique_bases = list(groups.keys())
    random.shuffle(unique_bases)
    # Split at beat level (e.g., 50 beats -> 40/10 for 0.8)
    split_idx = int(len(unique_bases) * train_ratio)
    split_idx = max(1, min(split_idx, len(unique_bases) - 1))
    train_bases = set(unique_bases[:split_idx])
    valid_bases = set(unique_bases[split_idx:])
    # Flatten grouped samples maintaining subset integrity
    train_samples = [s for base in train_bases for s in groups[base]]
    valid_samples = [s for base in valid_bases for s in groups[base]]

    print(f"Total samples: {len(samples)} (train: {len(train_samples)}, valid: {len(valid_samples)})")

    metadata_entries = []

    for subset_name, subset_samples in (("train", train_samples), ("valid", valid_samples)):
        output_subset_dir = output_dir / subset_name

        for idx, sample in enumerate(tqdm(subset_samples, desc=f"Processing {subset_name}")):
            audio = process_audio(sample, sample_rate)
            filename = f"{prefix}_{sample.base_name}_{sample.segment_idx:02d}_{idx:04d}.wav"
            destination = output_subset_dir / filename
            save_audio(audio, sample_rate, destination)

            metadata_entries.append(
                {
                    "audio": f"{subset_name}/{filename}",
                    "text": sample.text,
                }
            )

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata to {metadata_path}")
    print("Dataset preparation completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MusicGen fine-tuning dataset")
    parser.add_argument("--raw-metadata", type=Path, required=True, help="Path to raw metadata JSON file")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Path to directory containing segmented WAV files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for processed dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Target sampling rate (default: 32000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--prefix", type=str, default="beat", help="Prefix for generated filenames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dataset(
        raw_metadata=args.raw_metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        sample_rate=args.sample_rate,
        seed=args.seed,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
