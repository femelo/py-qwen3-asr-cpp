# Py-Qwen3-ASR-cpp

Python bindings for for the **Qwen3-ASR** and **Forced Aligner** as implemented by [qwen3-asr.cpp](https://github.com/predict-woo/qwen3-asr.cpp). Powered by a high-performance C++ backend. This library provides a seamless way to transcribe audio and align text with word-level timestamps using GGUF models.

## 🚀 Features

* **High-Level Wrapper**: Simple, pythonic API for transcription and forced alignment.
* **Automatic Audio Handling**: Built-in support for WAV files and automatic conversion of other formats (MP3, FLAC, etc.) via `ffmpeg`.
* **NumPy Integration**: Pass audio data directly as `np.float32` arrays.
* **GGUF Support**: Efficient model loading and inference.
* **Word-Level Timestamps**: Precise alignment of text to audio for subtitling or analysis.

---

## 📦 Installation

```bash
pip install py-qwen3-asr-cpp
```

**Note**: For non-WAV audio files, ensure `ffmpeg` is installed and available in your system `PATH`.

### 2. Usage Examples

## 🛠 Usage

### 1. Basic Transcription
Transcribe an audio file into text with just a few lines of code.

```python
from py_qwen3_asr_cpp.model import Qwen3ASRModel

# Initialize the model (it handles downloading if a repo ID is provided)
model = Qwen3ASRModel(
    asr_model="qwen3-asr-0.6b-q8-0",
    n_threads=4
)

# Transcribe from file
result = model.transcribe("audio.mp3")
print(f"Detected language: {result.language}")
print(f"Transcription: {result.text}")
```

### 2. Forced Alignment
Align a known text transcript to an audio file to obtain word-level timestamps.

```python
model = Qwen3ASRModel(
    asr_model="qwen3-asr-0.6b-q8-0",
    align_model="qwen3-forced-aligner-0.6b-q8-0"
)

# Text to align with the audio
text = "The quick brown fox jumps over the lazy dog"

alignment = model.align("audio.wav", text=text)

for word in alignment.words:
    print(f"Word: {word.word:12} | Start: {word.start}ms | End: {word.end}ms")
```

### 3. Pipeline and Configuration

### 3. Combined Pipeline
Transcribe and immediately align to get the best of both worlds in one call.

```python
asr_res, align_res = model.transcribe_and_align("interview.wav")

print(f"Full text: {asr_res.text}")
print(f"Total words aligned: {len(align_res.words)}")
```

## ⚙️ Configuration

The `Qwen3ASRModel` accepts several parameters to fine-tune performance:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `asr_model` | `str` | Path to ASR GGUF model or HuggingFace ID. |
| `align_model`| `str` | Path to Aligner GGUF model (optional). |
| `n_threads` | `int` | Number of CPU threads to use (default: 4). |
| `language` | `str` | Force a specific language (e.g., "en", "zh"). |
| `max_tokens` | `int` | Maximum tokens for the decoder. |
| `print_timing`| `bool` | Whether to print inference timing to stdout. |


## 📝 License

This project is licensed under the **Apache License 2.0**.

**Author:** femelo  
**Copyright:** © 2026

---

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
