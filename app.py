import gradio as gr
import torch
import scipy.io.wavfile
import numpy as np
import tempfile
import os
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    VitsModel,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Model loading — lazy, cached after first call
# ---------------------------------------------------------------------------

_whisper_processor = None
_whisper_model = None
_nllb_tokenizer = None
_nllb_model = None
_tts_model = None
_tts_tokenizer = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_whisper():
    global _whisper_processor, _whisper_model
    if _whisper_model is None:
        _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium"
        ).to(DEVICE)
    return _whisper_processor, _whisper_model


def load_nllb():
    global _nllb_tokenizer, _nllb_model
    if _nllb_model is None:
        _nllb_tokenizer = NllbTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M", src_lang="fra_Latn"
        )
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        ).to(DEVICE)
    return _nllb_tokenizer, _nllb_model


def load_tts():
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        _tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ewe")
        _tts_model = VitsModel.from_pretrained("facebook/mms-tts-ewe").to(DEVICE)
    return _tts_tokenizer, _tts_model


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def transcribe_french(audio_path: str) -> str:
    """French audio → French text via Whisper medium."""
    processor, model = load_whisper()
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        predicted_ids = model.generate(
            **inputs,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="fr", task="transcribe"),
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def translate_fr_to_ewe(french_text: str) -> str:
    """French text → Ewe text via NLLB-200-distilled-600M."""
    tokenizer, model = load_nllb()
    inputs = tokenizer(
        french_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    target_lang_id = tokenizer.convert_tokens_to_ids("ewe_Latn")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def synthesize_ewe(ewe_text: str) -> str:
    """Ewe text → Ewe audio WAV via Meta MMS-TTS-ewe. Returns path to WAV file."""
    tokenizer, model = load_tts()
    inputs = tokenizer(ewe_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**inputs).waveform
    wav = output.squeeze().cpu().numpy()
    # Normalize to int16
    wav_int16 = (wav / np.max(np.abs(wav)) * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    scipy.io.wavfile.write(tmp.name, rate=model.config.sampling_rate, data=wav_int16)
    return tmp.name


# ---------------------------------------------------------------------------
# Gradio interface handlers
# ---------------------------------------------------------------------------

def pipeline_from_audio(audio_path):
    """Full pipeline: French audio → Ewe audio."""
    if audio_path is None:
        return "Veuillez fournir un fichier audio.", "", None
    try:
        french_text = transcribe_french(audio_path)
        ewe_text = translate_fr_to_ewe(french_text)
        ewe_audio_path = synthesize_ewe(ewe_text)
        return french_text, ewe_text, ewe_audio_path
    except Exception as e:
        return f"Erreur: {e}", "", None


def pipeline_from_text(french_text):
    """Text shortcut: French text → Ewe audio (skips Whisper)."""
    if not french_text.strip():
        return "", None
    try:
        ewe_text = translate_fr_to_ewe(french_text)
        ewe_audio_path = synthesize_ewe(ewe_text)
        return ewe_text, ewe_audio_path
    except Exception as e:
        return f"Erreur: {e}", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
## Pipeline de localisation audio Français → Éwé
**Westland-corp × GLK 360**

Ce pipeline convertit automatiquement du contenu **audio ou texte en français** vers
de l'**audio en langue éwé** (~7 millions de locuteurs au Togo, Ghana et Bénin).

Il s'adresse aux institutions, ONG et radios communautaires qui produisent du contenu
en français et souhaitent le rendre accessible aux communautés éwéphones.

### Architecture — modèles open source
| Étape | Modèle | Rôle |
|-------|--------|------|
| 1 — Transcription FR | `openai/whisper-medium` | Audio français → Texte français |
| 2 — Traduction FR→EWE | `facebook/nllb-200-distilled-600M` | Texte français → Texte éwé |
| 3 — Synthèse vocale EWE | `facebook/mms-tts-ewe` | Texte éwé → Audio éwé |

> **Version actuelle :** modèles de base, sans fine-tuning spécifique à l'éwé.
> La qualité de traduction reflète l'état de l'art sur une paire de langues à faibles ressources.
> Une version fine-tunée sur corpus annoté est en développement.
"""

with gr.Blocks(title="FR→EWE Pipeline | Westland-corp × GLK 360") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab("Entrée audio (pipeline complet)"):
        gr.Markdown("### Étape 1 : Chargez un fichier audio en français")
        audio_input = gr.Audio(
            label="Audio français (MP3, WAV, M4A)",
            type="filepath",
        )
        btn_audio = gr.Button("Lancer le pipeline", variant="primary")
        with gr.Row():
            out_fr_text = gr.Textbox(label="Transcription française (Whisper)", lines=4)
            out_ewe_text_a = gr.Textbox(label="Traduction éwé (NLLB-200)", lines=4)
        out_ewe_audio_a = gr.Audio(label="Audio éwé synthétisé (MMS-TTS)", type="filepath")

        btn_audio.click(
            fn=pipeline_from_audio,
            inputs=[audio_input],
            outputs=[out_fr_text, out_ewe_text_a, out_ewe_audio_a],
        )

    with gr.Tab("Entrée texte (sans transcription)"):
        gr.Markdown("### Entrez directement du texte en français")
        text_input = gr.Textbox(
            label="Texte français",
            lines=4,
            placeholder="La santé est un droit fondamental pour tous les citoyens du Togo.",
        )
        btn_text = gr.Button("Traduire et synthétiser", variant="primary")
        out_ewe_text_b = gr.Textbox(label="Traduction éwé (NLLB-200)", lines=4)
        out_ewe_audio_b = gr.Audio(label="Audio éwé synthétisé (MMS-TTS)", type="filepath")

        btn_text.click(
            fn=pipeline_from_text,
            inputs=[text_input],
            outputs=[out_ewe_text_b, out_ewe_audio_b],
        )

    gr.Markdown("""
---
**Westland-corp** (Togo) · **GLK 360** (US)
Contact : info@glk360.com
""")

if __name__ == "__main__":
    demo.launch()
