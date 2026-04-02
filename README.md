# Machine Learning Pipeline: DVC + MLflow

## 📋 Project Overview

This project demonstrates a **complete machine learning pipeline** using modern tools for data version control and experiment tracking. We train a **Random Forest Classifier** to predict diabetes from the **Pima Indians Diabetes Dataset**.

### What Does This Project Do?

Think of this project as an automated factory for machine learning:
1. **Raw Data** → Preprocessing → **Clean Data**
2. **Clean Data** → Training → **Trained Model**
3. **Trained Model** → Evaluation → **Performance Reports**

The entire process is tracked, reproducible, and shareable with your team!

---

## 🎯 Key Concepts Explained

### 1. **What is DVC (Data Version Control)?**

**DVC** is like Git, but for machine learning projects. While Git tracks code changes, DVC tracks:
- **Data files** (large datasets)
- **Model files** (trained ML models)
- **Pipeline stages** (preprocessing, training, evaluation steps)

**Why is this important?**
- **Reproducibility**: Run the same code on the same data and get the same results
- **Collaboration**: Team members can work with the same data without storing huge files in Git
- **Tracking Changes**: Know exactly when your data or models changed

**How it works in our project:**
```
dvc.yaml → Describes pipeline stages (what to run and in what order)
params.yaml → Stores parameters (settings that control model behavior)
```

### 2. **What is MLflow?**

**MLflow** is an experiment tracking tool. When you train a machine learning model, many things change:
- Hyperparameters (settings like number of trees, tree depth)
- Performance metrics (accuracy, precision, recall)
- Data versions used
- Code versions used

**MLflow records all this information**, so you can:
- Compare different runs (which hyperparameters worked best?)
- Reproduce results (what settings gave us 95% accuracy?)
- Share experiments with your team

**In our project**, MLflow logs:
- Best hyperparameters found
- Accuracy score
- Confusion matrix (detailed performance breakdown)
- Classification report (precision, recall, F1-score)

### 3. **What is a Random Forest Classifier?**

A **Random Forest** is a machine learning model that makes decisions by:
1. Creating multiple "decision trees" (like a flowchart of yes/no questions)
2. Each tree votes on the answer
3. The most popular vote wins

**Example decision tree:**
```
Age > 30?
├─ YES → Blood Glucose > 120?
│        ├─ YES → Predict: DIABETES
│        └─ NO → Predict: NO DIABETES
└─ NO → Predict: NO DIABETES
```

Multiple trees together = **Stronger, more accurate predictions**

### 4. **Hyperparameters: What Are They?**

**Hyperparameters** are settings you choose BEFORE training the model. They control HOW the model learns.

Common hyperparameters for Random Forest:
- `n_estimators`: How many trees to create (e.g., 100, 200 trees)
- `max_depth`: How deep each tree can grow (deeper = more complex patterns)
- `min_samples_split`: Minimum samples needed to split a node (prevents overfitting)

**Finding the best hyperparameters** is like tuning a guitar:
- Too loose (few trees, shallow): Model is too simple, misses patterns
- Too tight (many trees, deep): Model memorizes data, performs poorly on new data
- Just right (balanced): Model learns real patterns and generalizes well

Our project uses **GridSearchCV** to automatically test different combinations!

### 5. **Training vs Testing Data**

We split our data into two parts:
- **Training Data (80%)**: Used to teach the model
- **Testing Data (20%)**: Used to evaluate if the model really learned

**Why split?** If you teach and test on the same data, the model just memorizes answers. We need to test on NEW data to see if it actually learned!

### 6. **Accuracy: Measuring Performance**

**Accuracy** = (Correct Predictions) / (Total Predictions) × 100%

Example: If we made 100 predictions and got 95 correct, accuracy = 95%

⚠️ **Important**: Accuracy alone can be misleading. Confusion matrix gives more details!

### 7. **Confusion Matrix: The Details**

Shows exactly what the model got right and wrong:

```
                Predicted Positive    Predicted Negative
Actual Positive      True Positive       False Negative
Actual Negative      False Positive      True Negative
```

This helps identify if the model is better at finding diabetes or finding healthy people.

---

## 📁 Project Structure

```
machinelearningpipeline/
├── dvc.yaml              # Pipeline configuration (stages to run)
├── params.yaml           # Model parameters (settings)
├── requirements.txt      # Python packages needed
├── .env.example          # Template for environment variables (DO NOT EDIT)
├── .env                  # Your actual credentials (NOT pushed to GitHub)
├── .gitignore            # Files to NOT track in Git
├── README.md             # This file
│
├── data/
│   ├── raw/
│   │   ├── data.csv                    # Original dataset
│   │   └── data.csv.dvc                # DVC metadata for tracking
│   └── processed/
│       └── data.csv                    # Cleaned data (output of preprocessing)
│
├── models/
│   └── model.pkl                       # Trained model (saved as pickle file)
│
└── src/                  # Source code scripts
    ├── __init__.py
    ├── preprocess.py     # Data cleaning script
    ├── train.py          # Model training script
    └── evaluate.py       # Model evaluation script
```

---

## 🔧 How Each Script Works

### 1. **preprocess.py** - Data Preparation
```
Input:  data/raw/data.csv (raw dataset)
↓
Clean and prepare data
↓
Output: data/processed/data.csv (ready for training)
```

**What it does:**
- Reads raw CSV file
- Cleans the data (removes headers, etc.)
- Saves processed data

### 2. **train.py** - Model Training
```
Input:  data/processed/data.csv (clean training data)
↓
1. Split into training (80%) and testing (20%)
2. Test many hyperparameter combinations
3. Find and train the best model
4. Save trained model
5. Log results to MLflow
↓
Output: models/model.pkl (trained model)
        MSflow: Best hyperparameters & accuracy
```

**Key steps:**
- **GridSearchCV**: Tests combinations of hyperparameters
  - n_estimators: [100, 200]
  - max_depth: [5, 10, None]
  - min_samples_split: [2, 5]
  - min_samples_leaf: [1, 2]

- **Cross-validation** (cv=5): Tests on 5 different data splits for reliability

- **Best model is selected** and saved as pickle file

### 3. **evaluate.py** - Model Testing
```
Input:  data/processed/data.csv (test data)
        models/model.pkl (trained model)
↓
Make predictions
Calculate accuracy
Log to MLflow
↓
Output: Accuracy report
```

---

## 🚀 Running the Pipeline

### **Option 1: Run Everything Automatically with DVC**
```powershell
# Install DVC if not already installed
pip install dvc

# Run the entire pipeline
dvc repro
```

This automatically:
1. Runs preprocessing if data changed
2. Trains model if preprocessing changed
3. Evaluates model if training changed
4. Skips steps that haven't changed (very efficient!)

### **Option 2: Run Individual Scripts**
```powershell
# Install dependencies
pip install -r requirements.txt

# Step 1: Preprocess data
python src/preprocess.py

# Step 2: Train model
python src/train.py

# Step 3: Evaluate model
python src/evaluate.py
```

---

## 📊 Monitoring Experiments with MLflow

### **View Experiment Results**
```powershell
# Start MLflow UI (opens in browser)
mlflow ui
```

Then visit: `http://localhost:5000` to see:
- All training runs
- Hyperparameters used
- Accuracy scores
- Confusion matrices
- Classification reports

### **Compare Runs**
You can see:
- Which hyperparameters gave highest accuracy?
- How did run #1 compare to run #2?
- Trends over multiple experiments

---

## 🔐 Security: Protecting Credentials

### **The Problem**
Our code uses MLflow to track experiments. MLflow needs credentials:
- Username
- Password/Token
- Tracking URI

❌ **NEVER hardcode credentials in Python files!** This exposes secrets to GitHub.

### **The Solution**
We use a `.env` file:

1. **`.env.example`** (safe to push to GitHub)
   - Contains only placeholders
   - Shows what variables are needed

2. **`.env`** (NEVER push to GitHub!)
   - Contains your actual credentials
   - Only on your local machine
   - Loaded by `load_dotenv()`

3. **Code loads from `.env`**
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Loads MLFLOW_TRACKING_URI, etc.
   ```

### **How to Set Up**
1. Copy `.env.example` to `.env`
2. Fill in your actual credentials in `.env`
3. Python loads these automatically when scripts run
4. `.env` is in `.gitignore` so it never goes to GitHub

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `scikit-learn` | Machine learning algorithms |
| `dvc` | Data versioning and pipeline |
| `mlflow` | Experiment tracking |
| `python-dotenv` | Load environment variables from .env |
| `pyyaml` | Parse YAML configuration files |

Install all: `pip install -r requirements.txt`

---

## 🎓 Learning Resources

### **DVC Concepts**
- **Pipeline**: A series of stages that transform data
- **Stage**: A single step (e.g., preprocessing)
- **Dependencies**: Inputs that a stage needs
- **Outputs**: Files a stage produces
- **Params**: Configuration values a stage uses

### **MLflow Concepts**
- **Experiment**: A group of related runs
- **Run**: A single training execution
- **Metric**: A performance measurement (accuracy)
- **Parameter**: A hyperparameter setting
- **Artifact**: Files logged (models, reports)

### **Machine Learning Concepts**
- **Features (X)**: Input variables (age, glucose, etc.)
- **Target (y)**: Output we predict (diabetes: yes/no)
- **Train/Test Split**: Dividing data for unbiased evaluation
- **Overfitting**: Model memorizes instead of generalizing
- **Cross-validation**: Testing on multiple data folds

---

## 🔄 Workflow Example

1. **Make a change** to hyperparameters in `params.yaml`
   ```yaml
   train:
     n_estimators: 150  # Changed from 100
   ```

2. **Run the pipeline**
   ```powershell
   dvc repro
   ```

3. **DVC detects** that params changed, so it:
   - Re-runs training with new params
   - Skips preprocessing (no changes)

4. **View results** in MLflow UI
   ```powershell
   mlflow ui
   ```

5. **Compare** new accuracy with previous runs

6. **Iterate** until you find the best parameters!

---

## 📝 For Adding New Stages to Pipeline

If you want to add a new stage (e.g., feature engineering):

```powershell
dvc stage add -n feature_engineering `
    -d src/feature_engineering.py `
    -d data/processed/data.csv `
    -p feature_engineering.threshold `
    -o data/features/features.csv `
    python src/feature_engineering.py
```

Then update `dvc.yaml` to define how the new stage connects to others.

---

## 🤝 Team Collaboration

**With DVC and MLflow:**
- Everyone uses the same data versions (no "my version works" problems)
- Everyone can see all experiments and results
- Remote storage (S3, DagsHub) keeps data synchronized
- Easy to understand what changed and why

---

## ✅ Checklist Before Pushing to GitHub

- ✅ `.env` is in `.gitignore` (credentials never pushed)
- ✅ `.env.example` has only placeholder values
- ✅ No hardcoded passwords in code
- ✅ `requirements.txt` lists all dependencies
- ✅ README explains the project clearly
- ✅ Code has helpful comments

---

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🎉 Happy DataScience-ing!

You now have a professional ML pipeline that tracks data, code, and experiments. Keep experimenting and iterating!

Questions? Check the comments in the Python files for detailed explanations of each step.
	
	
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py
	
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py