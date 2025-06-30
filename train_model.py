import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# ========== 1. Load data historis
df = pd.read_csv('data_cuaca_histori.csv')

# ========== 2. Fungsi normalisasi label sesuai mapping user
def normalize_condition(label):
    label = label.lower()
    if 'clear' in label or 'sunny' in label:
        return 'Clear'
    elif 'partly cloudy' in label or 'cloudy' in label:
        return 'Cloudy'
    elif 'overcast' in label:
        return 'Overcast'
    elif 'patchy rain' in label or 'light rain' in label:
        return 'Patchy Rain Possible'
    elif 'moderate rain' in label or (label.strip() == 'rain'):
        return 'Rain'
    elif 'heavy rain' in label:
        return 'Heavy Rain'
    elif 'thunderstorm' in label:
        return 'Thunderstorm'
    elif 'mist' in label or 'fog' in label:
        return 'Fog'
    elif 'snow' in label:
        return 'Snow'
    else:
        return 'Other'

df['condition_normalized'] = df['condition'].apply(normalize_condition)

# ========== 3. Analisa distribusi label setelah normalisasi
print("Distribusi label setelah normalisasi:")
print(df['condition_normalized'].value_counts())

# ========== 4. Siapkan fitur dan label
required_columns = ['temp_c', 'humidity', 'wind_kph', 'cloud', 'dewpoint_c', 'hour', 'precip_mm', 'pressure_mb', 'uv']
df = df.dropna(subset=required_columns)

X = df[required_columns]
y = df['condition_normalized']

# ========== 5. Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========== 6. Stratified split (supaya distribusi label di train/test seimbang)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ========== 7. Latih model
model = RandomForestClassifier(
    n_estimators=150, random_state=42, class_weight='balanced_subsample'
)
model.fit(X_train, y_train)

# ========== 8. Evaluasi
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
laporan = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
print("✅ Model dilatih dan disimpan.")
print(f"🎯 Akurasi: {akurasi:.2%}")
print("\n📋 Classification Report:\n", laporan)

# ========== 9. Simpan model dan encoder
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model_cuaca.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_order.pkl')

# ========== 10. Simpan akurasi ke file
with open('model/accuracy.txt', 'w') as f:
    f.write(str(round(akurasi * 100, 2)))

# ========== 11. Confusion Matrix (proportion)
cm = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
labels = le.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig('model/confusion_matrix_normalized.png')
plt.close()