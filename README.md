# AnthroScore & IntellScore Evaluation

This project evaluates AI-generated responses using two custom scoring metrics: **AnthroScore** and **IntellScore**. These metrics are designed to assess how *human-like* (anthropomorphic) and *informative* the responses are, based on emotional content, pronoun usage, length, and sentiment.

Originally developed as a Google Colab notebook, this script has been adapted for local use.

## ✨ What It Does

- **AnthroScore**: Measures how emotionally supportive and personal the response is by counting first-person pronouns, emotional keywords, and overall sentiment.
- **IntellScore**: Assesses the informativeness of a response using word count and sentiment polarity.
- Uses `TextBlob` for sentiment analysis.
- Ranks responses from most to least anthropomorphic.

## 📦 Dependencies

Install required packages:

```bash
pip install textblob
python -m textblob.download_corpora
