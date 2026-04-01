import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 1. Read and Clean the dataset
print("="*80)
print("STEP 1: READING AND CLEANING THE DATASET")
print("="*80)

# Load the dataset
try:
    df = pd.read_csv('loan_data_set.csv')
    print("\n✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: loan_data_set.csv file not found!")
    print("Please make sure the file is in the current directory.")
    exit()

# Display basic information
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Checking for missing values ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found!")

# Handle missing values
print("\n--- Handling Missing Values ---")
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        # For numerical columns, fill with median
        if df[column].isnull().sum() > 0:
            median_val = df[column].median()
            df[column].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{column}' with median: {median_val}")
    else:
        # For categorical columns, fill with mode
        if df[column].isnull().sum() > 0:
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)
            print(f"Filled missing values in '{column}' with mode: {mode_val}")

# Check if there are any remaining missing values
print(f"\nRemaining missing values: {df.isnull().sum().sum()}")

# Remove duplicates if any
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"\nRemoved {initial_shape[0] - df.shape[0]} duplicate rows")

# Handle outliers (optional - using IQR method for numerical columns)
print("\n--- Handling Outliers ---")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        print(f"Column '{col}': {len(outliers)} outliers detected")
        # Capping outliers
        df[col] = df[col].clip(lower_bound, upper_bound)

print("\n✅ Data cleaning completed!")
print(f"Final dataset shape: {df.shape}")

# 2. Prepare data for Machine Learning
print("\n" + "="*80)
print("STEP 2: DATA PREPARATION FOR MODELING")
print("="*80)

# Identify target variable (assuming 'loan_status' or similar is the target)
# Common column names for loan status: 'loan_status', 'default', 'approved', etc.
target_column = None
possible_targets = ['loan_status', 'default', 'approved', 'status', 'Loan_Status', 'is_default']

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    # If no common target column found, assume the last column is the target
    print("\n⚠️ Could not automatically identify target column.")
    print("Available columns:", list(df.columns))
    target_column = input("\nPlease enter the target column name: ")
    if target_column not in df.columns:
        print(f"❌ Error: '{target_column}' not found in dataset!")
        exit()

print(f"\n🎯 Target variable: '{target_column}'")

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")

# Encode categorical variables
print("\n--- Encoding Categorical Variables ---")
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
        print(f"Encoded column: '{column}'")

# Encode target variable if it's categorical
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    print(f"\nTarget variable encoded. Classes: {target_encoder.classes_}")

# Split the data
print("\n--- Splitting Data (80% train, 20% test) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features (important for Logistic Regression)
print("\n--- Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler")

# 3. Apply Logistic Regression Model
print("\n" + "="*80)
print("STEP 3: APPLYING LOGISTIC REGRESSION MODEL")
print("="*80)

# Create and train the model
print("\n--- Training Logistic Regression Model ---")
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handles imbalanced datasets
)
logistic_model.fit(X_train_scaled, y_train)

print("✅ Model training completed!")

# Make predictions
print("\n--- Making Predictions ---")
y_pred = logistic_model.predict(X_test_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

# Detailed classification report
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# Additional metrics
print("\n--- Additional Metrics ---")
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate additional metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Feature importance (coefficients)
print("\n--- Feature Importance ---")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logistic_model.coef_[0]
})
feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualization of feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['Feature'], 
         feature_importance.head(10)['Absolute_Coefficient'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Top 10 Feature Importances - Logistic Regression')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
✅ Data Cleaning Completed:
   - Handled missing values
   - Removed duplicates
   - Handled outliers
   - Final dataset shape: {df.shape}

✅ Model Selection: Logistic Regression
   - Suitable for binary classification problems
   - Works well with linear decision boundaries
   - Provides feature importance through coefficients

✅ Model Performance:
   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
   - F1-Score: {f1_score:.4f}
   - Confusion Matrix saved as 'confusion_matrix.png'
   - Feature Importance plot saved as 'feature_importance.png'
""")

print("\n🎉 Analysis completed successfully!")