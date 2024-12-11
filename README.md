# **Text Analysis on Indonesian Languages Using NusaX**

## **Project Overview**
This project explores text analysis tasks using the **NusaX dataset**, a multilingual corpus for Indonesian and Javanese languages. The focus is on two key NLP tasks:
1. **Machine Translation**: Translating between Indonesian-English and Indonesian-Javanese using pre-trained models.
2. **Sentiment Analysis**: Implementing classical and advanced machine learning models for sentiment classification of Indonesian text.

---

## **Features**
- **Machine Translation Models**:
  - [Helsinki-NLP/opus-mt-id-en](https://huggingface.co/Helsinki-NLP/opus-mt-id-en): Pre-trained model for Indonesian-English translation.
  - [IndoBenchmark/IndoBART](https://huggingface.co/indobenchmark/indobart): Pre-trained model for Indonesian-Javanese translation.
- **Sentiment Analysis Models**:
  - **Classical Models**: SVM, MLP, and Word2Vec embeddings.
  - **Pretrained Models**: VADER and RoBERTa.

---

## **Requirements**
Ensure you have the following libraries installed:
- `scikit-learn`
- `torch`
- `transformers`
- `nltk`
- `gensim`
- `matplotlib`

### **Installing Dependencies**
Run this in a cell within the Jupyter Notebook to install all dependencies:
```python
!pip install scikit-learn torch transformers nltk gensim matplotlib
```
## **Usage**

### **1. Dataset Preparation**
1. Download the **NusaX dataset** from its official repository.
2. Organize the dataset into:
   - **Machine Translation**: Pairwise datasets for Indonesian-English and Indonesian-Javanese.
   - **Sentiment Analysis**: Labeled datasets for sentiment classification in Indonesian.

### **2. Machine Translation**
Run the machine translation cells in the notebook to:
- Perform translations using the following models:
  - Indonesian-English: `Helsinki-NLP/opus-mt-id-en`.
  - Indonesian-Javanese: `IndoBenchmark/IndoBART`.

### **3. Sentiment Analysis**
Run the sentiment analysis cells in the notebook to:
- Train and evaluate the following models:
  - **Classical Models**: SVM, MLP.
  - **Transformer-Based Models**: RoBERTa and other pretrained models.

### **4. Visualizations**
- Visualize confusion matrices and accuracy trends for sentiment analysis models.
- Evaluate BLEU scores for machine translation tasks.

---

## **Results**

### **Machine Translation**
| **Model**                 | **Translation Task**       | **BLEU Score** |
|----------------------------|----------------------------|----------------|
| Helsinki-NLP/opus-mt-id-en | Indonesian to English      | 32.4           |
| IndoBenchmark/IndoBART     | Indonesian to Javanese     | 27.8           |

### **Sentiment Analysis**
| **Model**       | **Accuracy** | **Precision (Macro Avg)** | **Recall (Macro Avg)** | **F1-Score (Macro Avg)** |
|------------------|--------------|---------------------------|-------------------------|--------------------------|
| SVM             | 79%          | 0.77                      | 0.77                   | 0.77                     |
| MLP             | 77%          | 0.78                      | 0.75                   | 0.76                     |
| VADER           | 37.25%       | 0.16                      | 0.33                   | 0.19                     |
| Word2Vec + SVM  | 48%          | 0.43                      | 0.39                   | 0.39                     |

---


## **References**
1. Helsinki-NLP. *opus-mt-id-en*. Hugging Face. Retrieved from [https://huggingface.co/Helsinki-NLP/opus-mt-id-en](https://huggingface.co/Helsinki-NLP/opus-mt-id-en).
2. IndoBenchmark. *IndoBART*. Hugging Face. Retrieved from [https://huggingface.co/indobenchmark/indobart](https://huggingface.co/indobenchmark/indobart).
3. IndoNLP. *NusaX: Multilingual Parallel Corpus for Indonesian Local Languages*. GitHub. Retrieved from [https://github.com/IndoNLP/nusax](https://github.com/IndoNLP/nusax).


---

## **Future Improvements**
- Expand analysis to include additional local languages supported by NusaX.
- Fine-tune transformer models with a more diverse and larger dataset for improved sentiment classification.
- Optimize models for computational efficiency to enable broader deployment in resource-constrained environments.

---

## **Acknowledgments**
This project was made possible through the use of pre-trained models and datasets available on Hugging Face and the NusaX repository.
