import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # استبدال talib بـ pandas_ta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1️⃣ جلب بيانات EUR/USD كل 10 دقائق من آخر 60 يومًا
eurusd = yf.download("EURUSD=X", period="60d", interval="10m")

# 2️⃣ حساب المؤشرات الفنية باستخدام pandas_ta
eurusd['RSI'] = ta.rsi(eurusd['Close'], length=14)  # مؤشر القوة النسبية
macd = ta.macd(eurusd['Close'], fast=12, slow=26, signal=9)  # MACD
eurusd['MACD'] = macd['MACD_12_26_9']
eurusd['MACD_signal'] = macd['MACDs_12_26_9']
eurusd['SMA_50'] = ta.sma(eurusd['Close'], length=50)  # متوسط متحرك 50 فترة
eurusd['SMA_200'] = ta.sma(eurusd['Close'], length=200)  # متوسط متحرك 200 فترة
bbands = ta.bbands(eurusd['Close'], length=20)  # نطاقات بولينجر
eurusd['BB_upper'] = bbands['BBU_20_2.0']
eurusd['BB_middle'] = bbands['BBM_20_2.0']
eurusd['BB_lower'] = bbands['BBL_20_2.0']

# 3️⃣ تجهيز البيانات
eurusd['Future'] = eurusd['Close'].shift(-1)  # السعر المستقبلي
eurusd['Target'] = (eurusd['Future'] > eurusd['Close']).astype(int)  # 1 = ارتفاع، 0 = انخفاض
eurusd.dropna(inplace=True)

# اختيار الميزات
features = ['Close', 'RSI', 'MACD', 'MACD_signal', 'SMA_50', 'SMA_200', 'BB_upper', 'BB_middle', 'BB_lower']
X = eurusd[features]
y = eurusd['Target']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ تدريب نموذج الذكاء الاصطناعي XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train_scaled, y_train)

# 5️⃣ التقييم: حساب دقة النموذج
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 دقة النموذج: {accuracy * 100:.2f}%")

# 6️⃣ تنبؤ الصفقة القادمة
last_data = scaler.transform([X_test.iloc[-1]])
prediction = model.predict(last_data)

# 7️⃣ طباعة التوصية
if prediction[0] == 1:
    print("📈 تنبؤ: السعر سيرتفع - **ضع صفقة BUY (شراء)**")
else:
    print("📉 تنبؤ: السعر سينخفض - **ضع صفقة SELL (بيع)**")
