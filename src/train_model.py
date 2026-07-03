import os
import sqlite3
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance

def main():
    db_path = os.path.join("data", "processed", "olist.db")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load processed data from SQLite
    conn = sqlite3.connect(db_path)
    print("Reading processed_orders table from database...")
    df = pd.read_sql("SELECT * FROM processed_orders", conn)
    conn.close()
    
    # Extract order_month from purchase_ts
    df['purchase_ts'] = pd.to_datetime(df['purchase_ts'])
    df['order_month'] = df['purchase_ts'].dt.month
    
    # Select features and target
    categorical_cols = ['payment_type', 'product_category_name', 'customer_state', 'seller_state']
    numeric_cols = ['distance_km', 'price', 'freight_value', 'payment_installments', 'order_month']
    target_col = 'is_problem_order'
    
    # Fill missing product category name with 'unknown'
    df['product_category_name'] = df['product_category_name'].fillna('unknown')
    df['payment_type'] = df['payment_type'].fillna('unknown')
    
    # Select relevant rows
    df_model = df[categorical_cols + numeric_cols + [target_col]].dropna()
    print(f"Dataset size for modeling: {len(df_model)} rows")
    print(f"Class distribution:\n{df_model[target_col].value_counts(normalize=True)}")
    
    # One-hot encoding of categorical variables
    print("One-hot encoding categorical features...")
    X_cats = pd.get_dummies(df_model[categorical_cols], drop_first=True)
    X = pd.concat([df_model[numeric_cols], X_cats], axis=1)
    y = df_model[target_col]
    
    # Train-test split
    print("Splitting into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Handle class imbalance with SMOTE
    print("Applying SMOTE to balance classes in training set...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Balanced class distribution: {y_train_bal.value_counts()}")
    
    # Train XGBoost
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train_bal, y_train_bal)
    
    # Evaluate
    print("Evaluating model...")
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, preds_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Feature Importance Plot
    print("Saving feature importance chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_importance(model, max_num_features=10, ax=ax, importance_type='gain')
    plt.title('Top 10 Feature Importances (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'))
    plt.close()
    print("Feature importance chart saved to outputs/feature_importance.png")
    
    # Package and save the model
    model_package_path = os.path.join(outputs_dir, "risk_model_package.pkl")
    print(f"Packaging model and schema to {model_package_path}...")
    
    # Collect unique category values for mapping in Streamlit
    categories = {}
    for col in categorical_cols:
        categories[col] = sorted(df_model[col].unique().tolist())
        
    model_data = {
        'model': model,
        'feature_names': list(X.columns),
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'categories': categories,
        'auc': auc
    }
    
    with open(model_package_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model packaged successfully!")

if __name__ == "__main__":
    main()
