#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from audio_io import load_audio, save_wav_pcm16
from demucs.apply import apply_model
from model import load_model
from stitch import stitch_restored

SAMPLE_RATE = 44100


def separate(model, mixture, device, shifts=0, overlap=0.25, segment=None):
    ref = mixture.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    normed = ((mixture - mean) / std)[None].to(device)
    with torch.no_grad():
        out = apply_model(model, normed, shifts=shifts, overlap=overlap, segment=segment, device=device)
    out = (out * std + mean)[0].cpu().numpy()  # (n_sources, C, T)
    return {name: out[i].T for i, name in enumerate(model.sources)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="Input audio file (any format soundfile reads).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    ap.add_argument("--shifts", type=int, default=0, help="Random-shift ensembling (demucs.apply); 0 = off.")
    ap.add_argument("--overlap", type=float, default=0.25, help="Chunk overlap fraction for long clips.")
    ap.add_argument("--segment", type=float, default=None, help="Chunk length in seconds; default: model's own.")
    ap.add_argument("--stems-only", action="store_true", help="Skip writing restored.wav.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    wav, sr = load_audio(args.input, target_sr=SAMPLE_RATE)  # (T, 2)
    mixture = torch.from_numpy(np.ascontiguousarray(wav.T)).float()  # (2, T)

    estimates = separate(model, mixture, device, shifts=args.shifts, overlap=args.overlap, segment=args.segment)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, x in estimates.items():
        save_wav_pcm16(out_dir / f"{name}.wav", x, sr)
        print(f"  wrote {out_dir / f'{name}.wav'}")

    if not args.stems_only:
        restored, crossover_hz = stitch_restored(
            estimates["vocals"], estimates["music"], estimates["extension"], sr)
        save_wav_pcm16(out_dir / "restored.wav", restored, sr)
        print(f"  wrote {out_dir / 'restored.wav'} (content/extension crossover: {crossover_hz:.0f} Hz)")


if __name__ == "__main__":
    main()
