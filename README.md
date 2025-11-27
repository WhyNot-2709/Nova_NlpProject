Nova NLP: Medical Conversation Summarization & Ailment Suggestion

Abstractive medical dialogue summarization with downstream ailment inference.
Built using a fine-tuned encoder–decoder transformer, custom inference pipeline, regex-based hallucination mitigation, and domain-aligned evaluation.

Overview

Nova NLP converts raw doctor–patient conversations into concise, clinically structured summaries.
Instead of copying dialogue fragments, the system performs true abstraction: identifying symptoms, treatments, lifestyle recommendations, and follow-ups. It also suggests potential ailment categories based on extracted medical cues.

Motivation

Healthcare documentation is:

Time-intensive

Prone to omissions

Highly dependent on human note-taking skill

Difficult to standardize at scale

Most summarizers ignore medical context or hallucinate confidently. This project aims to reduce cognitive load for clinicians by generating accurate, reliable, well-structured SOAP-style summaries.

Problem Statement

Build an automated system that:

Summarizes long, informal medical conversations into clinically useful text.

Minimizes hallucinations and fabricated medical claims.

Suggests probable ailment categories without issuing diagnoses.

Key technical challenges:

Informal, fragmented conversational data

Domain-specific terminology

High hallucination risk in generative models

Evaluating summary quality beyond ROUGE

Dataset

Training/Validation: Hugging Face medical dialogue datasets

Additional Testing: 30 manually curated conversations from Kaggle, Hugging Face, and synthetic ChatGPT dialogues chosen for ambiguity and rare vocabulary

Designed to stress-test hallucination, coherence, and medical grounding

Proposed Pipeline
1. Data Ingestion

Load medical dialogues (JSON/CSV)

Standardize speaker labels

Preserve context ordering and timestamps

2. Preprocessing

Remove formatting noise

Tokenize and normalize case

Maintain clinical terminology consistency

3. Model Training

Fine-tuned pretrained encoder–decoder transformer
(T5/BART family chosen for stability + biomedical evidence)

Split into train, dev, dev2, tune, and final validation for controlled iteration

Mixed-precision + CUDA acceleration for efficient training

4. Inference Pipeline

Custom infer.py with:

Keyword extraction

Ailment categorization suggestions

Safety filtering

Summary restructuring

5. Hallucination Control

Regex patterns remove:

Fake drug names

Unsupported dosages

Invented diagnoses

Post-generation validation ensures summaries stay grounded in inputs

6. Evaluation

Started with ROUGE (structural but shallow)

Switched to BERTScore for semantic understanding, factual coherence, and domain relevance

Manual clinician-style qualitative scoring for reliability

Expected Outcomes

Concise, medically faithful conversation summaries

Reduced hallucination rate during inference

Lightweight ailment suggestion for downstream triaging or EMR systems

Reproducible, extensible research pipeline

Applications

Telemedicine documentation

Clinical workflow automation

Patient communication & discharge summaries

EMR data cleaning and structuring

Medical chatbot grounding and safety systems

Repository Contents

/src – model, dataloaders, training scripts

/infer.py – inference + hallucination control

requirements.txt

Report.pdf

Example model outputs

README (this file)

Tech Stack

Python

PyTorch

Hugging Face Transformers

CUDA for GPU acceleration

Pandas, Regex, TQDM