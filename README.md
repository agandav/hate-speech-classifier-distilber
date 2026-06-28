# Hate Speech & Harm Classifier (Fine-Tuned DistilBERT)

Fine-tuned DistilBERT to classify social media comments into five categories: Hate Speech, Violent, Offensive, Not Offensive, and Sarcasm. Trained on a combined dataset built from Davidson et al., HateXplain, and Sarcasm Headlines (not included in this repo — see citations below).

## What this project actually found

While benchmarking my model against three teammates' models and a human-labeled ground truth across 80 test comments, I found a specific, repeatable failure mode: my model's harm classification and its sentiment output sometimes contradicted each other on the same input. Statements like "go back to your country" were correctly flagged as Hate Speech, but didn't always carry the negative-sounding language a sentiment model expects — the model was reading tone, not content, and the two signals disagreed.

To address this, I built a VADER-based sentiment-conflict layer that uses disagreement between sentiment polarity and harm classification as its own signal, alongside leet-speak deobfuscation and pattern matching for common evasion tactics (e.g., "kys" → "kill yourself"). That layer recovered roughly 11% accuracy specifically on the failure mode it targeted.

## Results

- 85% accuracy on an 80-sample benchmark (5-category classification)
- Raw DistilBERT inference alone matched a baseline SVM model (74%) — the signal layer, not model size, drove the accuracy gain
- Inverse-frequency class weighting applied to correct a 77% class imbalance in training data

## What's in this repo

- `Sai_Gandavarapu_DistilBERT.ipynb` — full notebook: preprocessing, class weighting, fine-tuning, the signal layer, and an interactive demo cell

## What's not in this repo

Training datasets are not included. Sources:
- Davidson et al., *Automated Hate Speech Detection and the Problem of Offensive Language* (2017)
- HateXplain (Mathew et al., 2021)
- Sarcasm Headlines Dataset (Misra & Arora)

## Known limitations

This is a class project, not a production system. The benchmark size (80 samples) is small, and the sentiment-conflict layer reduces but does not eliminate cases where tone and content disagree.
