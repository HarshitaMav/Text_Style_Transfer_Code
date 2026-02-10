# ✨ Multi-Style Text_Style_Transfer_Code


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Model](https://img.shields.io/badge/Model-FLAN--T5--Base-green)

A unified controllable text rewriting system that converts a single input sentence into five stylistic variants using a single fine-tuned FLAN-T5 model.

---

## 🚀 Features

Generate text in 5 distinct styles:

- 🧑‍💼 **Professional**
- 🙂 **Casual**
- 🙏 **Polite**
- 🌐 **Social**
- 😄 **Emojify**


---

## 🧠 Model Overview

- Base Model: `google/flan-t5-base`
- Architecture: Encoder–Decoder (Seq2Seq)
- Parameters: ~250M
- Max Length: 128 tokens
- Training: 3 epochs with gradient accumulation
- Evaluation: ROUGE + BLEU

---

## 📊 Results

| Metric     | Score  |
|------------|--------|
| ROUGE-1    | 0.7358 |
| ROUGE-2    | 0.5923 |
| ROUGE-L    | 0.7247 |
| BLEU       | 0.5376 |

Strong semantic preservation with clear stylistic differentiation across 4/5 styles.

---

### Example:
### Input:
convert to professional: I am good. How are you?

### Output:
I am pleased to inform you that I am doing well.
