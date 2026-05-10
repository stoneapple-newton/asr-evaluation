# Metric Definitions

## Word Error Rate (WER)

```
WER = (S + D + I) / N
```

- `S` substitutions, `D` deletions, `I` insertions computed via Levenshtein alignment
- `N` = number of words in the reference
- Computed after lowercasing and whitespace normalization
- Implemented in `eval/metrics/text_metrics.py:wer()`

## Character Error Rate (CER)

Same formula applied character-by-character instead of word-by-word.
More sensitive for agglutinative languages or when punctuation matters.

## Missing Content Rate

```
missing_content_rate = D / max(1, N)
```

Fraction of reference words that were dropped by the hypothesis.
High values indicate the ASR model missed significant content.

## Hallucination Rate

```
hallucination_rate = I / max(1, |hyp_words|)
```

Fraction of hypothesis words that have no alignment in the reference.
High values indicate the ASR model added content not present in the audio.
A known failure mode for Whisper-family models in low-speech regions.

## Diarization Error Rate (DER)

```
DER = (False Alarm + Missed Detection + Confusion) / Total Reference Speech
```

Computed via `pyannote.metrics.diarization.DiarizationErrorRate` when available;
falls back to an interval-overlap approximation otherwise.
Implemented in `eval/metrics/diar_metrics.py:der_score()`.

## Real-Time Factor (RTF)

```
RTF = processing_seconds / audio_seconds
```

RTF < 1.0 means the pipeline runs faster than real time.
Implemented in `eval/metrics/latency_cost.py:rtf()`.

## Sentence Similarity Score (SSS)

Semantic similarity between ground-truth and hypothesis using cosine similarity
of Ollama embeddings over aligned text chunks. Range: 0.0–1.0, higher is better.
Implemented in `eval/metrics/sss_metrics.py:sss_score()`.

SSS complements WER by catching paraphrase errors that WER penalizes harshly
but that preserve semantic meaning.
