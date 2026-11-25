# Speech-to-Text, Speech Processing & Text-to-Speech

Study notes **Build Multimodal Generative AI Applications**, covering:

- Lecture: [Speech-to-Text Technologies](https://www.coursera.org/learn/build-multimodal-generative-ai-applications/lecture/8oQAW/speech-to-text-technologies)
- Reading: [Text Processing, Speech Processing, and Text-to-Speech](https://www.coursera.org/learn/build-multimodal-generative-ai-applications/supplement/2a92T/text-processing-speech-processing-and-text-to-speech)

---
Use Mistral and gTTS to Create Your Personal Storyteller

## Table of contents

1. [Overview](#overview)
2. [Speech-to-Text (STT)](#speech-to-text-stt)
   - [What it is](#what-it-is)
   - [How STT evolved](#how-stt-evolved)
   - [The STT pipeline](#the-stt-pipeline)
   - [End-to-end models](#end-to-end-models)
   - [Quick start code](#quick-start-code)
3. [Text processing](#text-processing)
4. [Speech processing](#speech-processing)
5. [Text-to-Speech (TTS)](#text-to-speech-tts)
6. [Putting it together: multimodal AI](#putting-it-together-multimodal-ai)
7. [Real-world applications](#real-world-applications)
8. [Challenges](#challenges)
9. [Future directions](#future-directions)
10. [Getting started](#getting-started)
11. [Glossary](#glossary)

---

## Overview

Multimodal AI systems need to handle human language in two forms: **written text** and **spoken speech**. Three connected technologies make this possible:

| Technology | Input → Output | Purpose |
| --- | --- | --- |
| Text processing (NLP) | Text → meaning | Understand and analyse written language |
| Speech processing / STT | Audio → text and meaning | Understand spoken language |
| Text-to-Speech (TTS) | Text → audio | Give AI a natural-sounding voice |

```
            ┌──────────────┐        ┌────────────────┐        ┌──────────────┐
  Speech ──▶│  Speech-to-  │──Text─▶│ Text processing │──Text─▶│ Text-to-     │──▶ Speech
            │  Text (ASR)  │        │   (NLP / LLM)   │        │ Speech (TTS) │
            └──────────────┘        └────────────────┘        └──────────────┘
```

---

## Speech-to-Text (STT)

### What it is

Speech-to-Text, also called **Automatic Speech Recognition (ASR)**, turns spoken language into written text. It combines audio signal processing with natural language understanding. Modern systems recognise speech patterns and **phonemes** (the smallest units of sound in a language), work across many languages, and adapt to different accents and speaking styles.

### How STT evolved

| Era | Approach | Notes |
| --- | --- | --- |
| Early | Template matching, rule-based systems | Very small vocabularies |
| Statistical | Hidden Markov Models (HMMs) | Could handle more natural, continuous speech |
| Deep learning | Neural networks → end-to-end neural architectures | Big jump in accuracy |
| Current | Self-supervised learning + transformers | Learn from huge amounts of *unlabelled* audio |

### The STT pipeline

```
Raw audio
   │
   ▼
1. Pre-processing ........ clean and normalise; noise reduction, voice activity detection (VAD)
   │
   ▼
2. Feature extraction .... waveform → spectrogram or MFCCs
   │
   ▼
3. Acoustic model ........ maps short audio frames (a few ms) → phonemes / sound units
   │
   ▼
4. Decoder + language model  phonemes / sub-words → words, using context to improve accuracy
   │
   ▼
5. Post-processing ....... punctuation, capitalisation, formatting
   │
   ▼
Text output
```

### End-to-end models

Traditional systems keep the acoustic model, language model and decoder as separate parts. **End-to-end** models such as **Wav2Vec2** (Facebook AI Research) map audio directly to text in a single neural network. They are pre-trained on thousands of hours of audio and then fine-tuned for speech recognition. This makes the pipeline simpler and more accurate.

Typical steps:

1. Load the audio file.
2. Pre-process it into the format the model expects (for example, 16 kHz mono).
3. Pass it through the model.
4. Decode the output into a transcription.

### Quick start code

> These are example snippets written for these notes. They are not copied from the lecture.

**Wav2Vec2 with Hugging Face Transformers**

```bash
pip install torch transformers librosa
```

```python
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)

# 1. Load audio and resample to 16 kHz
speech, sr = librosa.load("sample.wav", sr=16000)

# 2. Pre-process
inputs = processor(speech, sampling_rate=16000, return_tensors="pt")

# 3. Run the model
with torch.no_grad():
    logits = model(inputs.input_values).logits

# 4. Decode to text
predicted_ids = torch.argmax(logits, dim=-1)
print(processor.batch_decode(predicted_ids)[0])
```

**OpenAI Whisper (open source)**

```bash
pip install -U openai-whisper   # also needs ffmpeg installed
```

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("sample.wav")
print(result["text"])
```

---

## Text processing

Text processing lets AI understand written language.

**Natural Language Processing (NLP)** is the foundation. It allows AI to:

- analyse sentence structure and grammar
- understand context and meaning
- identify entities (people, places, things)
- extract key information from documents
- generate human-like responses

**Text classification** sorts text into predefined groups, for example:

- spam detection
- sentiment analysis
- topic categorisation
- content moderation

**Information extraction** pulls structured data out of unstructured text:

- named entity recognition (NER)
- relationship extraction
- key phrase identification
- summarisation

---

## Speech processing

Speech processing converts spoken language into text and works out what it means.

| Area | What it does | Examples |
| --- | --- | --- |
| Speech recognition | Audio → text | Accents and dialects, background noise, many languages, real-time transcription |
| Speaker recognition | Who is speaking? | Voice authentication, speaker identification ("who said what"), emotion detection |
| Speech understanding | What does it mean? | Command recognition, intent detection, context, dialogue management |

---

## Text-to-Speech (TTS)

TTS converts written text into natural-sounding speech.

**Features of modern TTS systems**

- neural TTS models that sound human-like
- emotional expression
- natural prosody (rhythm, stress) and intonation
- multiple voice options
- real-time synthesis

**Applications**

- virtual assistants
- audiobook narration
- accessibility tools
- interactive voice response (IVR) systems
- educational content

---

## Putting it together: multimodal AI

| Use case | What the combined system can do |
| --- | --- |
| Conversational AI | Accept text or speech, reply in text or speech, keep context across modalities, hold a natural dialogue |
| Content creation | Write content, create audio narration, produce multimedia presentations and interactive experiences |
| Accessibility | Read text aloud for people with visual impairments, transcribe speech for people with hearing impairments, multimodal learning materials, universal design |

---

## Real-world applications

| Field | STT / speech / TTS uses |
| --- | --- |
| Accessibility | Captions for videos and live events, screen readers |
| Virtual assistants | Understanding spoken commands, replying by voice |
| Healthcare | Medical transcription and documentation, patient communication |
| Education | Automated note-taking, language learning, accessible materials |
| Business | Meeting transcription, customer service, document analysis |
| Legal | Automated court reporting, deposition transcription |
| Entertainment | Interactive games, virtual characters |

---

## Challenges

**Technical**

- **Background noise** lowers accuracy and needs advanced filtering.
- **Speaker variability:** different voices, accents and dialects.
- **Real-time processing** must be optimised for live use.
- **Domain adaptation:** specialist vocabulary in fields such as medicine or law.
- **Low-resource languages** often lack enough training data.
- **Meaning beyond words:** understanding context and nuance.
- **Consistency across modalities** in multimodal systems.

**Ethical**

- privacy of voice data
- bias in language models
- accessibility for all users
- responsible AI development

---

## Future directions

- **Self-supervised learning:** train on large amounts of unlabelled audio, so fewer expensive transcriptions are needed.
- **Multilingual and multimodal models:** one system that handles many languages.
- **Better contextual understanding:** interpreting meaning, not just words.
- **Personalisation:** adapting to an individual's voice and preferences.
- **Edge / on-device processing:** better privacy and lower latency.
- **More natural and expressive TTS.**
- **Deeper integration with other modalities** (vision, video, and more).

---

## Getting started

1. Try open-source tools such as **OpenAI Whisper**.
2. Experiment with cloud services such as **Google Cloud Speech-to-Text** and **Azure AI Speech**.
3. Start with simple transcription projects.
4. Learn the basics of audio and signal processing.
5. Study some linguistics, especially phonetics and phonology.
6. Join practitioner communities: forums, workshops and hackathons.

---

## Glossary

| Term | Meaning |
| --- | --- |
| **ASR** | Automatic Speech Recognition; another name for STT |
| **STT** | Speech-to-Text |
| **TTS** | Text-to-Speech |
| **NLP** | Natural Language Processing |
| **Phoneme** | Smallest unit of sound in a language |
| **Spectrogram** | Visual representation of sound frequencies over time |
| **MFCC** | Mel-Frequency Cepstral Coefficients; compact audio features used by speech models |
| **VAD** | Voice Activity Detection; finds the parts of audio that contain speech |
| **HMM** | Hidden Markov Model; statistical model used in earlier ASR systems |
| **Acoustic model** | Maps audio features to sound units (phonemes) |
| **Language model** | Predicts likely word sequences to improve accuracy |
| **Prosody** | Rhythm, stress and intonation of speech |
| **Wav2Vec2** | Self-supervised, end-to-end speech model from Facebook AI Research |
| **Whisper** | Open-source multilingual speech recognition model from OpenAI |
| **Edge computing** | Running models on the device instead of in the cloud |

---
