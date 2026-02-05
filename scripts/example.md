# Welcome to mic-drop

This is a **Markdown** script that demonstrates the `.md` input feature.

## What this does

mic-drop will *automatically* strip the formatting from this file before
sending it to Tortoise TTS.  Everything you see here — headers, bold, links,
lists — is removed so the synthesiser only receives plain spoken text.

## Features

- Local processing — no cloud APIs
- Tortoise TTS for base speech generation
- RVC voice conversion with a custom `.pth` model
- Support for both `.txt` and `.md` input files

## Usage

Run it like any other script:

```
python -m tts_pipeline -i scripts/example.md -o output/example.wav -m models/voice.pth
```

That is all there is to it.  Enjoy!
