# Movie Genre Classification

An AI-powered system for classifying movie genres based on plot descriptions using TF-IDF vectorization and machine learning models.

---

## 📖 Project Overview

This project implements a **movie genre classification** pipeline that:
- Loads and preprocesses a dataset of movie plots and metadata
- Transforms text descriptions into TF-IDF feature vectors
- Trains and evaluates two models (Logistic Regression and Naive Bayes)
- Merges predictions with ground truth and reports performance metrics
- Visualizes model performance and data distributions


---

## 🚀 Functionality

1. **Data Loading**
   - Reads training and test data from text and CSV files
   - Parses movie ID, title, genre, and description fields

2. **Preprocessing**
   - Strips whitespace, lowercases text labels
   - Extracts release year from titles
   - Cleans and vectorizes descriptions with TF-IDF (unigrams + bigrams, stopwords removal)

3. **Model Training**
   - Trains **Logistic Regression** and **Multinomial Naive Bayes** on TF-IDF vectors

4. **Prediction & Evaluation**
   - Predicts genres on test set
   - Merges with true labels, computes accuracy, classification reports, and confusion matrices

5. **Visualization**
   - Confusion matrices for each model
   - Kernel density plots for release-year distribution (train vs. test)
   - Word-frequency bar charts (via TF-IDF scores)


---

## ✅ Code Quality & Best Practices

- **Modular functions** for data loading and preprocessing
- **Consistent naming** and clear variable scopes
- **Vectorized operations** with Pandas for speed
- **Use of `sklearn` pipelines** for reproducibility
- **Error handling** and data-cleaning before merges
- **Documentation** via comments and docstrings


---

## 💡 Innovation & Creativity

- Combining **title + description** for richer context
- Use of **bigrams** in TF-IDF to capture phrase-level meaning
- **Year extraction** from titles to explore temporal trends
- Dual-model comparison enables ensemble decision-making potential


---

## 📝 Documentation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Movie_Genre_Classification.git
   cd Movie_Genre_Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis script**:
   ```bash
   python movie_genre_analysis.py
   ```
   - Outputs accuracy scores, classification reports, and visualizations

4. **Experiment** with alternative models or parameters by editing `model` section in the script.


---

## 🛠 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk (for optional text preprocessing)


---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request with:
- Bug fixes or improvements
- New visualization ideas
- Integration of deep learning models (e.g., BERT)


---

 

