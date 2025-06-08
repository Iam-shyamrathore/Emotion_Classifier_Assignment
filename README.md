# Emotion Classifier Assignment

## Overview

This project implements an **Emotion Classifier** to predict emotions from text data using machine learning, presented in a Jupyter Notebook (`Emotion_Classifer_assignment.ipynb`). The classifier is trained on a dataset of labeled emotional texts and deployed as an interactive web application using Streamlit and ngrok. The project achieves a **test accuracy of 88.85%** and supports six emotion classes: sadness, joy, love, anger, fear, and surprise.

## Dataset

- **Source**: Hugging Face dataset (downloaded as `train-00000-of-00001.parquet`, likely `dair-ai/emotion` or similar)
- **Description**: Contains ~18,000 text samples labeled with one of six emotions (sadness, joy, love, anger, fear, surprise)
- **Splits**:
  - **Training**: ~16,000 samples (combined train and validation sets)
  - **Test**: 2,000 samples
- **Format**: Parquet file, processed using pandas
- **Label Distribution**: Approximately balanced, with sadness and joy being the most frequent classes (visualized in the notebook)

## Approach Summary

The Jupyter Notebook follows a structured pipeline to build and deploy the emotion classifier:

### 1. Data Loading and Exploration
- Loaded the dataset from a Parquet file using pandas
- Explored dataset characteristics with visualizations (e.g., bar plot for emotion counts, histogram for text lengths)

### 2. Preprocessing and Feature Engineering
- **Text Cleaning**: Removed URLs, hashtags, mentions, punctuation, and extra whitespace using `clean_text` function
- **Stopword Removal**: Used sklearn's `ENGLISH_STOP_WORDS` and filtered short words using `remove_stopwords`
- **Feature Extraction**: Applied TF-IDF vectorization (5,000 features, unigrams, and bigrams) using `TfidfVectorizer`

### 3. Model Training and Evaluation
- Trained multiple models (Random Forest selected as best model)
- **Validation Performance**: ~88% accuracy
- **Test Performance**: **88.85% accuracy**
- Generated confusion matrix, classification report, and error analysis with five misclassified examples
- Analyzed feature importance for Random Forest, listing the top 20 features

### 4. Deployment
- Built an interactive **Streamlit web app** (`emotion_app.py`) for real-time emotion prediction
- Features include:
  - Text input area
  - Example texts
  - Preprocessing details
  - Plotly bar chart for emotion probabilities
- Deployed using **ngrok** to create a public URL for remote access

## Dependencies

To run the notebook, ensure the following Python libraries are installed:

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.22.0
scikit-learn==1.5.0
streamlit==1.38.0
pyngrok==7.1.6
```

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit pyngrok
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install individually as listed above.



### 3. Run the Notebook
1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook Emotion_Classifer_assignment.ipynb
   ```
2. Execute all cells sequentially to load data, train the model, save artifacts, and launch the Streamlit app via ngrok

### 4. Access the Streamlit App
1. After running the final cell, a public URL will be printed (e.g., `https://<random>.ngrok-free.app`)
2. Open the URL in a browser to use the emotion classifier

## Key Features

- **Preprocessing**: Robust text cleaning and TF-IDF feature extraction
- **Model**: Random Forest classifier with **88.85% test accuracy**
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score), confusion matrix visualization, and error analysis
- **UI**: Interactive Streamlit app with probability visualization and example texts
- **Deployment**: Accessible via ngrok for remote use

## Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **88.85%** |
| **Validation Accuracy** | ~88% |
| **Features** | 5,000 TF-IDF features |
| **Classes** | 6 emotions (sadness, joy, love, anger, fear, surprise) |
## Screenshots
![image](https://github.com/user-attachments/assets/deb6f7cf-c778-4c83-8d8d-a3b3d2f464db)


![image](https://github.com/user-attachments/assets/a09d4202-d59a-48e2-ae09-a95ea3e829b8)


![image](https://github.com/user-attachments/assets/6376332a-419a-4aaa-97dd-e793de72b2bb)

![image](https://github.com/user-attachments/assets/aa8727d6-cee0-417d-9a0c-e21b9494a40d)



