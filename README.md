# Movie Genre Classification

An AI-powered system for classifying movie genres based on plot descriptions using TF-IDF vectorization and machine learning models.

---

## 📖 Project Overview

This project implements a **movie genre classification** pipeline that:

- Loads and preprocesses a dataset of movie plots and metadata.
- Transforms text descriptions into TF-IDF feature vectors (unigrams + bigrams, stopwords removal).
- Trains and evaluates two models: **Logistic Regression** and **Multinomial Naive Bayes**.
- Merges predictions with ground truth and reports performance metrics (accuracy, classification report, confusion matrix).
- Visualizes model performance and data distributions (confusion matrices, KDE plots).

---

## 🚀 Functionality

1. **Data Loading & Parsing**
   - Training data (`train_data.txt`): `id`, `title`, `genre`, `description`.
   - Test data (`test_data.txt`): `id`, `title`, `description`.
   - Solution data (`test_data_solution.txt`): `id`, `title`, `genre`, `description` (ground truth).

2. **Preprocessing**
   - Strip whitespace and lowercase the `genre` labels.
   - Extract release `year` from movie titles via regex (e.g., `"Movie Name (2010)"`).
   - Clean text: remove HTML tags, non-alphabetic characters, extra spaces, and stopwords (optional via NLTK).

3. **Feature Extraction**
   - **Combine** `title` + `description` for richer context.
   - **TF-IDF Vectorization**:
     ```python
     vectorizer = TfidfVectorizer(
         max_features=20000,
         stop_words='english',
         ngram_range=(1,2),
         min_df=2
     )
     X_train = vectorizer.fit_transform(train_df['text'])
     X_test  = vectorizer.transform(test_df['text'])
     ```

4. **Model Training & Prediction**
   - **Label Encoding** of genres:
     ```python
     label_encoder = LabelEncoder()
     y_train = label_encoder.fit_transform(train_df['genre'])
     ```
   - **Logistic Regression**:
     ```python
     lr = LogisticRegression(max_iter=10000)
     lr.fit(X_train, y_train)
     y_pred_lr = lr.predict(X_test)
     ```
   - **Multinomial Naive Bayes**:
     ```python
     nb = MultinomialNB()
     nb.fit(X_train, y_train)
     y_pred_nb = nb.predict(X_test)
     ```

5. **Evaluation**
   - **Merge** predictions with `test_data_solution` on `id`:
     ```python
     merged = pd.merge(df_solution[['id','genre']], test_df[['id','Predicted_Genre']], on='id')
     ```
   - **Accuracy**:
     ```python
     accuracy_score(merged['genre'], merged['Predicted_Genre'])
     ```
   - **Classification Report**:
     ```python
     print(classification_report(merged['genre'], merged['Predicted_Genre'], target_names=label_encoder.classes_))
     ```
   - **Confusion Matrix** (only labels present):
     ```python
     labels = np.unique(merged[['genre','Predicted_Genre']].values.ravel())
     cm = confusion_matrix(merged['genre'], merged['Predicted_Genre'], labels=labels)
     ConfusionMatrixDisplay(cm, display_labels=labels).plot(...)
     ```

6. **Visualization**
   - **Confusion Matrices** for both models with clear labels and rotated ticks.
   - **KDE Plot** for movie release `year` distribution (train vs. test):
     ```python
     sns.kdeplot(train_df['year'], label='Train', fill=True)
     sns.kdeplot(test_df['year'], label='Test', fill=True)
     ```
   - **Top TF-IDF Terms** bar chart by summed TF-IDF scores.

---

## ✅ Code Quality & Best Practices

- **Modular design**: Separate functions for loading, preprocessing, training, and evaluation.
- **Clear variable names** and inline comments for readability.
- **Reproducibility**: Fixed random seeds and consistent train/test splits.
- **Error handling**: Checks for missing data and mismatched IDs before merges.
- **Version control**: Well-structured commits, `.gitignore` for dependencies and data files.

---

## 💡 Innovation & Creativity

- **Dual-model comparison** enables easy benchmarking and potential ensemble strategies.
- **Text enrichment** by combining title and description.
- **N-gram features** (bigrams) capture multi-word expressions (e.g., "space ship").
- **Temporal analysis** via year extraction and KDE plots.

---

## 📝 Documentation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anitripathi/Movie_Genre_Classification.git
   cd Movie_Genre_Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   nltk.download('stopwords')  # optional
   ```

3. **Run the analysis**:
   ```bash
   python movie_genre_analysis.py
   ```

4. **Review outputs**:
   - Console: accuracy scores and classification reports.
   - Plots: confusion matrices and KDE distribution graphs.

---

## 🛠 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk (for stopwords)

---

## 🤝 Contributing

Contributions welcome! Please:
- Fork the repo
- Create a feature branch
- Submit pull requests with clear descriptions and tests

---
## by ANIVESH TRIPATHI

