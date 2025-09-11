#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, subprocess, tempfile
import torch
os.environ["TRANSFORMERS_PREFER_SAFETENSORS"] = "1"
from transformers import MarianMTModel, MarianTokenizer
os.environ["TRANSFORMERS_PREFER_SAFETENSORS"] = "1"
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ensure_ffmpeg():
    """Check ffmpeg availability."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        eprint("❌ ffmpeg non trovato. Installalo e assicurati che sia nel PATH (es. `brew install ffmpeg` su macOS, `sudo apt install ffmpeg` su Ubuntu)." )
        sys.exit(1)

def extract_audio(input_video: str, out_wav: str, sr: int = 16000):
    """Extract mono WAV 16kHz using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-ac", "1", "-ar", str(sr), "-vn",
        "-f", "wav", out_wav
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def transcribe_whisper(wav_path: str, model_size: str = "medium", device: str = None, compute_type: str = None):
    """
    Transcribe with faster-whisper.
    Returns: (full_text, detected_lang)
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        eprint("❌ Il pacchetto `faster-whisper` non è installato. Esegui: pip install faster-whisper")
        sys.exit(1)

    # Heuristic defaults
    if device is None:
        # Try CUDA, then MPS, else CPU
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    if compute_type is None:
        compute_type = "int8_float16" if device in ("cuda", "mps") else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(wav_path, beam_size=5)
    parts = []
    for seg in segments:
        parts.append(seg.text.strip())
    full_text = " ".join(parts).strip()
    detected_lang = getattr(info, "language", None) or "auto"
    return full_text, detected_lang

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import os

from deep_translator import GoogleTranslator
import time

def translate_to_italian(text: str, src_lang: str = None, **kwargs):
    """
    Traduzione via web (Google Translate) -> italiano.
    Evita modelli pesanti; zero problemi di 'meta tensor'.
    """
    translator = GoogleTranslator(source='auto', target='it')

    # Limita la dimensione dei chunk per evitare errori lato servizio.
    chunks = split_into_chunks(text, max_chars=4000)
    out = []
    for i, ch in enumerate(chunks, 1):
        # piccoli retry in caso di rete ballerina
        for attempt in range(3):
            try:
                out.append(translator.translate(ch).strip())
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.0 + attempt)  # backoff leggero
    return "\n".join(out).strip()

def split_into_chunks(text: str, max_chars: int = 2000):
    """Split text on sentence boundaries where possible, otherwise hard-wrap."""
    import re
    sentences = re.split(r'(?<=[\.!\?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(s) <= max_chars:
                cur = s
            else:
                # Hard wrap very long sentence
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Trascrivi un video e traduci in italiano.")
    ap.add_argument("input_video", help="Percorso al file video (es. input.mp4)")
    ap.add_argument("--out", default=None, help="File di output .txt con traduzione italiana")
    ap.add_argument("--whisper_model", default="medium", help="Modello faster-whisper (tiny, base, small, medium, large-v2, large-v3)")
    ap.add_argument("--device", default=None, help="Dispositivo: cpu | cuda | mps (default: auto)")
    ap.add_argument("--compute_type", default=None, help="Tipo calcolo per faster-whisper (es. int8, int8_float16)")
    ap.add_argument("--keep_audio", action="store_true", help="Non cancellare il WAV temporaneo")
    args = ap.parse_args()

    if not os.path.isfile(args.input_video):
        eprint(f"❌ File non trovato: {args.input_video}")
        sys.exit(1)

    ensure_ffmpeg()

    base = os.path.splitext(os.path.basename(args.input_video))[0]
    out_txt = args.out or f"{base}_tradotto_it.txt"

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, base + ".wav")
        print("🎬 Estrazione audio...")
        extract_audio(args.input_video, wav_path)

        print("🗣️  Trascrizione con faster-whisper...")
        transcript, detected_lang = transcribe_whisper(
            wav_path, model_size=args.whisper_model, device=args.device, compute_type=args.compute_type
        )
        print(f"🌐 Lingua rilevata: {detected_lang}")

        print("🌍 Traduzione in italiano (transformers M2M100)...")
        translated = translate_to_italian(transcript, src_lang=detected_lang)

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(translated.strip() + "\n")

        if args.keep_audio:
            # Save the wav next to the txt for inspection
            keep_path = os.path.abspath(base + "_audio.wav")
            subprocess.run(["cp", wav_path, keep_path], check=True)
            print(f"💾 Audio WAV salvato: {keep_path}")

    print(f"✅ Traduzione salvata: {os.path.abspath(out_txt)}")

if __name__ == "__main__":
    main()
