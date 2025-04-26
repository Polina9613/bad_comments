import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
import re
import string

nltk.download('punkt')
nltk.download('stopwords')

snowball = SnowballStemmer('russian')
russian_stop_words = stopwords.words('russian')

def preprocess_text(text, remove_stopwords=True, stem_words=True):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text, language='russian')
    
    if remove_stopwords:
        words = [w for w in words if w not in russian_stop_words]
    
    if stem_words:
        words = [snowball.stem(w) for w in words]
    
    return ' '.join(words)

df = pd.read_csv('labeled.csv')
df['toxic'] = df['toxic'].apply(int)
df['cleaned_text'] = df['comment'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], 
    df['toxic'], 
    test_size=0.2, 
    random_state=42
)

def train_classical_ml(models):
    results = {}
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()
    
    return pd.DataFrame(results).T

classical_models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'NaiveBayes': MultinomialNB(),
    'RandomForest': RandomForestClassifier(class_weight='balanced'),
    'XGBoost': XGBClassifier(scale_pos_weight=np.sum(y_train == 0)/np.sum(y_train == 1)),
    'CatBoost': CatBoostClassifier(verbose=0, class_weights=[1, np.sum(y_train == 0)/np.sum(y_train == 1)])
}

classical_results = train_classical_ml(classical_models)

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

lr_params = {
    'tfidf__max_features': [5000, 10000, 15000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear']
}

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_params,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_
y_pred = lr_best.predict(X_test)

rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(class_weight='balanced'))
])

rf_params = {
    'tfidf__max_features': [10000, 15000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 50, 100],
    'clf__min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_params,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

final_results = classical_results.copy()

final_results.loc['LogReg_Optimized'] = {
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted')
}

final_results.loc['RF_Optimized'] = {
    'precision': precision_score(y_test, y_pred_rf, average='weighted'),
    'recall': recall_score(y_test, y_pred_rf, average='weighted'),
    'f1': f1_score(y_test, y_pred_rf, average='weighted')
}

plt.figure(figsize=(10, 6))
final_results['f1'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Model Comparison by F1 Score')
plt.xlabel('F1 Score')
plt.grid(axis='x')
plt.tight_layout()
plt.show()