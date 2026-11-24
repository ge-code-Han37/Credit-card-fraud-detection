
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('credit_card_fraud_messsy.csv')


categorical_cols = [
    'merchant_category',
    'payment_channel',
    'card_type',
    'device_type',
    'transaction_country',
    'home_country'
]

df = pd.get_dummies(df, columns=categorical_cols, dtype=int)


df = df.drop(columns=[
    'transaction_id',
    'customer_id',
    'transaction_datetime'
])


X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))

