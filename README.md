# AnthroScore & IntellScore Evaluation

This project was built from scratch to evaluate AI-generated responses based on how **human-like** (AnthroScore) and **informative** (IntellScore) they are. It uses custom scoring functions, sentiment analysis, and basic NLP techniques to analyze and rank chatbot-style responses.

## 🧠 What It Does

- **AnthroScore** evaluates emotional tone and personal language (e.g., first-person pronouns, emotional keywords, sentiment).
- **IntellScore** scores informativeness based on word count and sentiment quality.
- Responses are ranked by anthropomorphic traits for analysis and comparison.
- Built entirely in Python using `TextBlob`, `nltk`, and `pandas`.

## 📦 Dependencies

Install required libraries:

```bash
pip install textblob pandas nltk
python -m textblob.download_corpora

