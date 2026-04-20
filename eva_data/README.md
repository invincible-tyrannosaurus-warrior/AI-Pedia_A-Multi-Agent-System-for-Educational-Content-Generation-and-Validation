# eva_data README

This file explains the role of each subfolder inside `eva_data` and clarifies the relationship between `student_lessons` and `evaluation_release`.

## Overview

`eva_data` is the main workspace for this teaching-content evaluation project. It contains:

- original or intermediate processing artifacts
- evaluation inputs prepared for the "student models"
- the final evaluation code and organized result package

If you only want the **final package for paper writing, code upload, and citation**, go directly to:

- `evaluation_release/`

---

## Subfolder guide

### 1. `assets/`
Stores resource files related to teaching-content generation or presentation.

Typical use cases:
- images
- figures
- visual assets used during lesson or slide generation

This folder mainly belongs to the **content-generation stage**, not the final evaluation release.

### 2. `generated_code/`
Stores automatically generated or experimentally produced code.

Typical use cases:
- data-processing scripts
- lesson/slide generation scripts
- auxiliary experiment code

This folder contains **process artifacts**, not the final release package.

### 3. `slides/`
Stores slide-level representations of the teaching content or closely related intermediate structures.

Typical use cases:
- slide structures for lessons
- titles, content blocks, and visual summaries

This is one of the key intermediate layers between the original teaching materials and the final evaluation-ready lesson format.

### 4. `uploads/`
Stores uploaded source files or other external input materials.

Typical use cases:
- originally imported content files
- user-uploaded materials

This folder is closer to the **source input layer**.

### 5. `vector_store/`
Stores vector-retrieval or embedding-related data.

Typical use cases:
- content indexing
- retrieval support
- similarity-search caches

It supports content organization and retrieval, but it is not a final evaluation-results folder.

### 6. `student_lessons/`
This is one of the **most important evaluation-input directories**.

It contains lesson + quiz data prepared for the student-model evaluations. Each topic folder typically contains:
- `student_input.json`

It also includes:
- `manifest.json`

This directory functions as the **direct dataset input** to the evaluation pipeline. For each topic it provides:
- `lesson_text`
- `slides`
- `questions`
- gold answers

In other words, `student_lessons/` is:
**the evaluation-ready teaching-and-quiz dataset prepared for the student models**.

### 7. `evaluation_artifacts/`
This is a local archive of intermediate experiment outputs and retrieved result folders.

It contains:
- pre/post evaluation results for different models
- intermediate copies of the evaluation pipeline

It is best understood as a **working archive**, not the final publication-oriented package.

### 8. `evaluation_release/`
This is the most important folder for downstream use.

It is the compact **final release package** intended for:
- paper writing
- code upload
- experiment archiving

It contains:
- `README.md`
- `code/`: evaluation pipeline code
- `results/`: final evaluation outputs
- `supporting_files/`: support files, diagnostics, and metadata
- `results_summary.csv`
- `results_summary.md`

**If you need a folder for external sharing, upload, or citation, use `evaluation_release/`.**

---

## How the teaching content was preprocessed

To obtain the contents inside `student_lessons/`, the teaching materials were not passed to the models in raw form. Instead, they were reorganized into a structure specifically designed for evaluation.

### Main preprocessing goal
The goal was to convert the teaching materials into a format that gives the student models the best chance to perform well and consistently. In particular, the preprocessing was designed to:

- **maximize the chance that the models can benefit from the content**
- **stabilize model behavior during evaluation**
- **reduce noise from raw file formatting**

### Core preprocessing logic
Based on the final `student_input.json` structure, the teaching content was transformed as follows:

1. **Topic-based organization**
   - each lesson is stored under one topic folder
   - this makes per-topic evaluation and analysis straightforward

2. **Unified lesson representation**
   - the content is serialized into `lesson_text`
   - a `slides` structure is also preserved
   - titles, body text, formulas, code snippets, and chart summaries are merged into a consistent JSON schema

3. **Unified quiz representation**
   - each topic is paired with a fixed set of multiple-choice questions
   - each question includes:
     - question text
     - four options
     - the gold answer
     - an explanation

4. **Manifest-based indexing**
   - `manifest.json` records all topics and corresponding input files
   - this allows the evaluation pipeline to iterate over the entire dataset in a reproducible way

### Why this preprocessing helps the student models
This structure improves evaluation quality in several ways:

- it reduces formatting noise from the original materials
- it avoids forcing the models to first understand arbitrary file structures
- it gives different models the **same standardized input format**
- it improves the interpretability of pre-learn vs. post-learn comparisons
- it makes the relationship between lesson content, questions, and answers easier to trace

In short, `student_lessons/` is a **standardized lesson-and-quiz layer designed to maximize and stabilize student-model performance during evaluation**.

---

## Why these models were selected

The chosen models were selected to cover multiple capability levels within the small / mini-model regime and to compare how models at different strengths respond to pre-learn and post-learn evaluation.

### Main evaluated models
- `microsoft/Phi-3-mini-4k-instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### Reasons for selecting them

#### 1. They are all small or mini instruct models
This matches the project setup of treating the model as a "student model" or "rookie student."

#### 2. They cover multiple capability levels
- TinyLlama: very small and relatively weak, useful as a low-capability baseline
- Qwen 1.5B / 3B: same family, different scales, useful for scaling comparisons
- Phi-3 mini / Phi-3.5 mini: same family, different generation levels, useful for upgrade comparisons

#### 3. They can all be deployed locally in the same server environment
These models fit well into the Runpod + RTX 4090 + local Transformers setup, which helps control experimental variables.

#### 4. They are suitable for testing whether lesson injection really helps
This model set spans a wide enough capability range to reveal:
- which models benefit strongly from teaching materials
- which models are already strong and therefore improve only slightly
- which models are too weak to make effective use of the lesson content

---

## The relationship between `student_lessons` and `evaluation_release`

This is the most important structural distinction in the entire `eva_data` folder.

### `student_lessons/` is the input layer
It provides:
- lesson content
- quiz questions
- gold answers
- dataset structure

So it is the **evaluation dataset layer**.

### `evaluation_release/` is the final output and release layer
It provides:
- how the evaluation was run (code)
- what the evaluation produced (results)
- how those results should be interpreted (README, summaries, supporting files)

So it is the **final package for paper writing, result citation, and code release**.

### A simple way to remember the distinction
- `student_lessons/` = **the standardized teaching dataset fed into the evaluation pipeline**
- `evaluation_release/` = **the final organized package built from running code on that dataset**

### Practical rule
If you want:
- the lesson materials and quiz inputs -> look in `student_lessons/`
- the final code and evaluation outputs -> look in `evaluation_release/`

---

## Most important practical takeaway

If you need material for:
- paper writing
- code upload
- supplementary material preparation

start with:

- `evaluation_release/`

That directory is already organized for release use, while `student_lessons/` is the core input dataset layer that the release package depends on.
