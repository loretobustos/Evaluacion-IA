import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Corpus representativo de funciones seguras y vulnerables
data = {
    'code': [
        "void copy(char *src) { char dest[10]; strcpy(dest, src); }",
        "void copy_safe(char *src) { char dest[10]; strncpy(dest, src, sizeof(dest)-1); dest[9]='\\0'; }",
        "char *query = malloc(256); sprintf(query, 'SELECT * FROM users WHERE id = %s', user_input);",
        "sqlite3_prepare_v2(db, 'SELECT * FROM users WHERE id = ?', -1, &res, 0);",
        "int ptr_free(char *p) { free(p); free(p); }",
        "int ptr_clean(char *p) { if(p) { free(p); p = NULL; } }",
        "void gets_read() { char buf[20]; gets(buf); }",
        "void fgets_read() { char buf[20]; fgets(buf, sizeof(buf), stdin); }",
        "int calc(int b) { int a = 100 / b; return a; }",
        "int calc_safe(int b) { if(b == 0) return 0; return 100 / b; }",
        "system(user_command);",
        "execve(valid_binary, args, env);"
    ] * 50,
    'is_vulnerable': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 50
}
df = pd.DataFrame(data)
print(f"Dimensiones del dataset cargado: {df.shape}")
df.head()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['code'], df['is_vulnerable'], test_size=0.30, random_state=42, stratify=df['is_vulnerable']
)

# Vectorizador con preservación de operadores y unigramas + bigramas
vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b|[^\w\s]', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
print(f"Vocabulario generado: {X_train.shape[1]} características sintácticas.")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Entrenamiento completado exitosamente.")

y_pred = model.predict(X_test)
print("--- REPORTE DE EVALUACIÓN ---")
print(classification_report(y_test, y_pred, target_names=['Seguro', 'Vulnerable']))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Seguro', 'Vulnerable'],
            yticklabels=['Seguro', 'Vulnerable'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.show()

