from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Veri kümesini yükleyin (ör. iris veri kümesi)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM sınıflandırıcısını oluşturma
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Modeli eğitme
svm.fit(X_train_scaled, y_train)

# Tahminlerde bulunma
y_pred = svm.predict(X_test_scaled)

# Doğruluk skorunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluk skoru: {accuracy:.2f}")