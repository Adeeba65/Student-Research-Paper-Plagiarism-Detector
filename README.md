# Student-Research-Paper-Plagiarism-Detector

**Domain:** Education
**GUI Library:** Tkinter (Python's built-in library — no installation needed)
**AI Algorithm:** TF-IDF Vectorization + Cosine Similarity, plus N-gram Shingle Matching for phrase-level highlighting (all implemented from scratch, no external ML libraries)

## What This Project Does

A desktop app that checks a student's research paper for plagiarism against a
library of existing reference papers.

1. **Add Reference Paper** — Build your "database" of existing papers/articles
   (paste title + full text). Saved permanently to `reference_papers.csv`.
2. **Reference Library** — View every paper currently stored, with word counts.
3. **Check Plagiarism** — Paste a new student paper. The app:
   - Converts every paper into a **TF-IDF vector** (a way of representing text
     as numbers based on how important each word is)
   - Calculates **Cosine Similarity** between the student's paper and every
     reference paper (a value from 0-100% showing how alike they are)
   - Ranks all reference papers by similarity score
   - **Highlights the exact overlapping phrases** in yellow so you can see
     precisely which sentences were copied (using 6-word phrase matching)
   - Gives a verdict: 🟢 Low / 🟠 Moderate / 🔴 High plagiarism risk

## How TF-IDF + Cosine Similarity Works (Simple Explanation)

- **TF (Term Frequency):** How often a word appears in a document
- **IDF (Inverse Document Frequency):** Words that appear in *every* document
  (like "the", "is") are less meaningful than rare, specific words — IDF
  down-weights common words and up-weights distinctive ones
- **TF-IDF Vector:** Each paper becomes a list of numbers representing how
  important each word is to that specific paper
- **Cosine Similarity:** Measures the angle between two papers' vectors —
  the smaller the angle, the more similar the papers are (100% = identical
  wording pattern, 0% = completely unrelated)

This is the same foundational concept used by real-world tools like Turnitin
and Google's document comparison features.

## How to Run (in VS Code)

### Requirements
- Python 3.8+ (Tkinter comes built-in — nothing to install)

### Steps
1. Open the `plagiarism_detector` folder in VS Code
2. Open a terminal (Terminal → New Terminal)
3. Run:
   ```bash
   python main.py
   ```

## How to Use It

1. Go to **"Add Reference Paper"** and add 2-3 sample papers/articles (can be
   real paragraphs from Wikipedia, textbooks, or old assignments — paste the
   full text)
2. Go to **"Check Plagiarism"**, paste a new paper (try copying a few sentences
   from one of your reference papers plus adding some original sentences, to
   see it in action)
3. Click **"Check for Plagiarism"** — see the similarity scores, the ranked
   table of matches, and the highlighted overlapping phrases



## Note on How AI Was Used

AI (Claude) was used to:
- Understand and implement TF-IDF and Cosine Similarity from scratch in plain
  Python (no scikit-learn/numpy), so the project has zero external dependencies
- Learn how n-gram "shingle" matching works for highlighting overlapping phrases
- Structure the multi-page Tkinter GUI (sidebar navigation, card layout, tables)
- Debug and test the similarity scores against sample text to confirm the
  algorithm behaves correctly (identical papers score ~100%, unrelated papers
  score near 0%, partially-copied papers score in between)

The domain choice, algorithm choice (TF-IDF over simpler word-counting), and
overall structure were reviewed and understood before finalizing.
