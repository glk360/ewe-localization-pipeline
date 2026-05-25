---
title: FR-EWE Audio Localization Pipeline
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: true
license: mit
---

# FR→EWE Audio Localization Pipeline

**Westland-corp (Togo) × GLK 360 (US)**

An open-source, end-to-end pipeline that converts French audio or text into Ewe audio.
Ewe is spoken by ~7 million people across Togo, Ghana, and Benin — with no production-grade
French-to-Ewe audio pipeline available as of 2026.

---

## The gap this fills

FR↔EWE text translation tools exist. No end-to-end audio pipeline does.

This pipeline adds the full audio layer: French speech in → Ewe speech out — built for
institutional deployment by NGOs, community radio stations, and health organizations that
produce French content and need to reach Ewe-speaking communities without costly human dubbing.

---

## Architecture

| Stage | Model | Role |
|-------|-------|------|
| 1 — FR Transcription | `openai/whisper-medium` | French audio → French text |
| 2 — FR→EWE Translation | `facebook/nllb-200-distilled-600M` | French text → Ewe text |
| 3 — EWE Speech Synthesis | `facebook/mms-tts-ewe` | Ewe text → Ewe audio |

All models are open source (MIT / CC-BY-NC). This release uses baseline models without
fine-tuning. A fine-tuned version trained on an annotated FR-EWE corpus is in development.

---

## Live demo

[HuggingFace Space — glk360/fr-ewe-pipeline](https://huggingface.co/spaces/glk360/fr-ewe-pipeline)

---

## Run locally

```bash
git clone https://github.com/glk360/ewe-localization-pipeline
cd ewe-localization-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

First run downloads ~3 GB of model weights. CPU-only is supported; GPU accelerates inference
if available.

---

## Status

**v0.1 — May 2026.** Baseline pipeline, proof-of-architecture.
Translation quality reflects the state of the art on a low-resource language pair.
Fine-tuning on the annotated corpus begins in Phase 1.

---

## Roadmap

- [ ] Annotated corpus: 500 FR-EWE pairs — corpus-v0.0 (CC-BY 4.0, to be published on HuggingFace)
- [ ] NLLB-200 fine-tune on corpus-v0.0 (Phase 1)
- [ ] MMS-TTS fine-tune on native Ewe recordings (Phase 1)
- [ ] FastAPI inference service (Phase 2)
- [ ] Expand to Kabiyé and Fon (Phase 3)

---

## License

Code: MIT
Model weights: see upstream licenses (Whisper MIT, NLLB-200 CC-BY-NC, MMS CC-BY-NC)

---

**Westland-corp** (Togo) · **GLK 360** (US)
Contact: info@glk360.com
