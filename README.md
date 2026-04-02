# Sentence Splitter
### *Small project for the Multilingual NLP course - Rémi Fouchérand*

---

**Main Idea:** Fine-tune a small classification head on top of a pretrained multilingual encoder to detect where a sentence must end. Not a very original approach but more simple to implement than manual feature-engineering.

Thus, this projects implements:
1. Data processing: Tokenization of the labeled dataset + mapping which token corresponds to a sentence ending.
2. Training of the classification head to predict if the current token is a sentence ending or not.
3. Quick evaluation of the model's performance.

## 1. Methodology

Without going into detail, here is a quick overview of the choices that were made to build this classifier.

- **Model Selection**: We chose BERT multilingual base model for the sake of simplicity. We could probably find a better performing one.

- **What to train?** We chose to **only train the token classification head** (the base model is freezed) with the given dataset, mainly because my PC can't handle large computations in a reasonable amount of time.

- **Dealing with class imbalance:** We chose Binary CrossEntropy as our loss function, **weighted by each class ocurence ratio** to counter the enormous class imbalance in the dataset (way more tokens that are NOT ending a sentence). For our validation loop we use **F1-score** as the preffered metric, way more informative than just accuracy this case.

- **Mitigating False positives:** As further detailed in the notebook, our model is really prone to FP. To help mitigate that we decided to move the decision boudary from 0.5 to 0.7 to help eliminate the tokens with weaker positive probability. 

Other smaller methodological choices are justified in the code comments

## 2. Results

Goal was to beat Spacy, our final model barely achieves that on the provided test set.

| Metric | Our Model (Threshold 0.7) | spaCy (Statistical) |
| :--- | :---: | :---: |
| **Precision** | 0.79 | 0.96 |
| **Recall** | 0.97 | 0.76 |
| **F1-Score** | 0.87 | 0.85 |
| **Macro Avg F1** | 0.93 | 0.92 |

---

**Short analysis** (a more detailed one is provided in the inference notebook)

Spacy shows a higher precision when our model is more prone to false positives (splitting where it shouldn't). Indeed our model tends to be very conservative and thus outperforms Spacy in recall. TDLR: Spacy misses a lot of sentence endings, our model splits too much.
Overall, our model is still more balanced getting a highr F1-score. As detailed further in the notebook, this is likely due to the nature of the dataset used and maybe Spacy would outperform in another context.

Still, theses results are honorable and i am satisfied with it. They could probabily be improved by unfreezing the base model and not just training the head.

## 3. Reproducing the results

First, clone this repo.

Then everything happens inside the ```inference.ipynb``` notebook which installs dependencies, loads the saved models weights and runs all the evaluations. Everything should run smoothly (i hope...)

If needed, one can train the model again by simply running the ```finetuning.py``` script.

---

**Notice to professor:** I uploaded the full dataset here for the sake of simplicity, if this is a privacy issue please tell me and i will remove it promptly. 