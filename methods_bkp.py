from src_modules.libraries import *

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english")) - {"no", "not"}
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text, lemmatize=False, stem=False):
        if not isinstance(text, str):
            text = str(text) if pd.notnull(text) else ""                                # Ensure all values in the 'text' column are strings
        
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r'\b(?:http|https|www|href)\b\S*', '', text)
        text = re.sub(r"@\\w+", "", text)
        text = re.sub(r"@\w+|#\w+", '', text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = re.findall(r"[a-zA-Z']+", text)
        tokens = [word for word in tokens if word not in self.stop_words]

        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(word, pos='v') for word in tokens]
        elif stem:
            tokens = [self.stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

# Create a default instance for easy use
_preprocessor = TextPreprocessor()
preprocess_text = _preprocessor.preprocess_text
    


class TextModelEvaluator:
    def __init__(self, text_columns, label_column="emotion", max_features=4500, ngram_range=(1, 2), static_state=42):
        self.text_columns = text_columns
        self.label_column = label_column
        self.static_state = static_state
        self.rand_state = random.randrange(1, 1000**3)
        self.vectorizer_settings = dict(max_features=max_features, ngram_range=ngram_range)
        self.base_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'SVM': LinearSVC(class_weight='balanced'),
            'Naive Bayes': MultinomialNB(),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=self.rand_state),
            'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=self.rand_state)
        }
        self.results = []
        self.best_model = None
        self.best_vectorizer = None
        self.best_variant = None
        self.best_model_name = None

    def compare_models(self, train_df, cross_validate=False):
        for variant_name, column in self.text_columns.items():
            X = train_df[column]
            y = train_df[self.label_column]

            vectorizer = TfidfVectorizer(**self.vectorizer_settings)
            X_tfidf = vectorizer.fit_transform(X)

            for model_name, model in self.base_models.items():
                if cross_validate:
                    scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy')
                    mean_score = float(f"{scores.mean():.6f}")
                else:
                    model.fit(X_tfidf, y)
                    y_pred = model.predict(X_tfidf)
                    mean_score = round(accuracy_score(y, y_pred), 6)

                self.results.append({
                    'Variant': variant_name,
                    'Model': model_name,
                    'Accuracy': mean_score
                })

        best_result = max(self.results, key=lambda r: r['Accuracy'])
        self.best_variant = best_result['Variant']
        self.best_model_name = best_result['Model']
        self.best_model = self._initialize_model(self.best_model_name)
        self.best_vectorizer = TfidfVectorizer(**self.vectorizer_settings)

        return self._format_results()

    def tune_best_model(self, train_df, param_grid):
        X = train_df[self.text_columns[self.best_variant]]
        y = train_df[self.label_column]
        X_tfidf = self.best_vectorizer.fit_transform(X)

        grid = GridSearchCV(self.best_model, param_grid, cv=10, scoring='accuracy')
        grid.fit(X_tfidf, y)
        self.best_model = grid.best_estimator_
        return grid.best_params_, grid.best_score_

    def evaluate_on_validation(self, train_df, val_df, plot_confusion=False, plot_roc=False):
        X_train = train_df[self.text_columns[self.best_variant]]
        y_train = train_df[self.label_column]
        X_val = val_df[self.text_columns[self.best_variant]]
        y_val = val_df[self.label_column]

        X_train_tfidf = self.best_vectorizer.fit_transform(X_train)
        X_val_tfidf = self.best_vectorizer.transform(X_val)

        self.best_model.fit(X_train_tfidf, y_train)
        y_pred = self.best_model.predict(X_val_tfidf)
        report = classification_report(y_val, y_pred, output_dict=True)

        metrics = {
            'Accuracy': float(f"{accuracy_score(y_val, y_pred):.6f}"),
            'Macro F1': float(f"{report['macro avg']['f1-score']:.6f}"),
            'Weighted F1': float(f"{report['weighted avg']['f1-score']:.6f}"),
            'Macro Precision': float(f"{report['macro avg']['precision']:.6f}"),
            'Macro Recall': float(f"{report['macro avg']['recall']:.6f}")
        }

        if plot_confusion:
            ConfusionMatrixDisplay.from_estimator(self.best_model, X_val_tfidf, y_val)
            plt.title(f"Confusion Matrix - {self.best_model_name} ({self.best_variant})")
            plt.show()

        if plot_roc:
            try:
                classes = sorted(train_df[self.label_column].unique())
                y_val_bin = label_binarize(y_val, classes=classes)
                model = OneVsRestClassifier(self.best_model)
                model.fit(X_train_tfidf, y_train)
                y_score = model.decision_function(X_val_tfidf) if hasattr(model, "decision_function") else model.predict_proba(X_val_tfidf)

                for i, class_label in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.2f})")

                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve: {self.best_model_name}")
                plt.legend()
                plt.show()
            except Exception as e:
                print(f"ROC curve failed: {e}")

        return metrics

    def train_final_model(self, train_df):
        column = self.text_columns[self.best_variant]
        X_train = train_df[column]
        y_train = train_df[self.label_column]
        X_train_tfidf = self.best_vectorizer.fit_transform(X_train)
        self.best_model.fit(X_train_tfidf, y_train)

    def predict(self, test_df):
        column = self.text_columns[self.best_variant]
        X_test = test_df[column]
        X_test_tfidf = self.best_vectorizer.transform(X_test)
        return self.best_model.predict(X_test_tfidf)

    def _initialize_model(self, name):
        if name.startswith("kNN"):
            k = int(name.split("-")[1].strip())
            return KNeighborsClassifier(n_neighbors=k)
        else:
            return self.base_models[name]

    def _format_results(self):
        results_df = pd.DataFrame(self.results)
        results_df['Accuracy Rank'] = results_df['Accuracy'].rank(method='dense', ascending=False).astype(int)
        return results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

      
        


        