# TikTok Video Claim Classification

A machine learning project that classifies TikTok videos as **claims** or **opinions**. The goal is to help prioritize videos for human review by identifying content that is more likely to contain a claim.

The project follows the **PACE framework** (Plan, Analyze, Construct, Execute) and compares two tree-based classification models: Random Forest and XGBoost.

## Project objective

TikTok receives user reports about videos that may violate its Terms of Service. Reviewing every reported video manually can create a significant moderation backlog.

This project builds a binary classification model using the `claim_status` variable:

| Class | Encoded value | Meaning |
|---|---:|---|
| Opinion | `0` | The video offers an opinion |
| Claim | `1` | The video makes a claim |

The analysis prioritizes **recall** because missing a video that contains a claim (a false negative) is considered more costly than sending an opinion video to human review (a false positive). The model is intended to support prioritization, not replace human moderation.

## Dataset

The notebook expects a file named `tiktok_dataset.csv`.

The dataset used in the notebook contains **19,382 rows and 12 columns before removing missing values**. The analysis identifies 298 rows with missing values and drops those rows, leaving **19,084 observations** for feature engineering and modeling. No duplicate observations were found.

### Columns

| Column | Description |
|---|---|
| `#` | Row identifier; excluded from modeling |
| `claim_status` | Target: `claim` or `opinion` |
| `video_id` | Video identifier; excluded from modeling |
| `video_duration_sec` | Video duration in seconds |
| `video_transcription_text` | Text transcription of the video |
| `verified_status` | Whether the author is verified |
| `author_ban_status` | Author moderation/ban status |
| `video_view_count` | Number of video views |
| `video_like_count` | Number of likes |
| `video_share_count` | Number of shares |
| `video_download_count` | Number of downloads |
| `video_comment_count` | Number of comments |

The target distribution is approximately balanced in the cleaned dataset:

- Claims: **50.35%**
- Opinions: **49.65%**

## Project workflow

The notebook is organized around the PACE stages.

### 1. Plan

- Define the business and modeling objective.
- Identify false positives and false negatives.
- Select recall as the primary model-selection metric because false negatives are considered more costly.

### 2. Analyze

- Inspect the dataset, data types, summary statistics, and missing values.
- Remove rows with missing values.
- Check for duplicate observations.
- Review the target class distribution.
- Explore the relationship between transcription length and claim status.

The notebook reports an average transcription length of approximately:

- Claims: **95.38 characters**
- Opinions: **82.72 characters**

### 3. Construct

- Create `text_length` from the character count of `video_transcription_text`.
- Encode `claim_status` as a binary target (`opinion = 0`, `claim = 1`).
- Remove the identifier columns `#` and `video_id`.
- One-hot encode `verified_status` and `author_ban_status`.
- Split the data into training, validation, and test sets using a **60% / 20% / 20%** ratio.
- Use stratification to preserve the target class proportions.
- Apply `CountVectorizer` to the transcription text using 2- and 3-word n-grams, English stop words, and a maximum of 15 features. The vectorizer is fitted on the training data only, then reused to transform validation and test data.

### 4. Execute

Train and compare two classifiers:

**Random Forest**
- Uses `GridSearchCV` with 5-fold cross-validation.
- Evaluates accuracy, precision, recall, and F1-score.
- Selects the best estimator using recall (`refit="recall"`).

**XGBoost**
- Uses `XGBClassifier` with `GridSearchCV` and 5-fold cross-validation.
- Evaluates the same classification metrics.
- Also selects the best estimator using recall.

The notebook compares validation-set confusion matrices and classification reports, selects Random Forest as the champion model, evaluates it on the held-out test set, and plots its feature importances.

## Model results

The saved notebook outputs report the following cross-validation results:

| Model | Best cross-validation recall | Precision at best estimator |
|---|---:|---:|
| Random Forest | 0.9957 | 1.0000 |
| XGBoost | 0.9887 | 0.9979 |

The notebook selects **Random Forest** as the champion model because it achieved the higher cross-validation recall.

The saved validation classification reports show rounded accuracy, precision, recall, and F1-scores of approximately 1.00 for both models. The saved final Random Forest test report also displays 1.00 for these metrics when rounded to two decimal places.

> **Important:** These are the results recorded in the notebook's saved outputs. The displayed test metrics are rounded, so `1.00` does not necessarily mean that every prediction was correct. Re-run the notebook to reproduce the results in your environment.

## Feature importance and interpretation

The notebook examines the Random Forest feature importances and reports that engagement-related variables are among the most predictive features. These include video views, likes, shares, downloads, and comments.

This finding should be interpreted carefully. Engagement values may accumulate after a video is published and may not be available at the time moderation prioritization is required. If a feature is recorded after the event the model is intended to predict, it can introduce **temporal or target leakage** and make offline performance look better than real-world performance.

Before deployment, confirm that every predictor is available at the exact decision time. Consider evaluating a second model that excludes post-publication engagement and moderation-status variables. The model should assist human reviewers rather than make moderation decisions automatically.

## Technologies and libraries

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost

## Installation and setup

### 1. Clone or download the project

Place the notebook and `tiktok_dataset.csv` in the same working directory.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `TikTok_Project_Lab.ipynb` and run the cells from top to bottom.

The notebook looks for `tiktok_dataset.csv` in the current working directory. When run in Google Colab, it also provides a file-upload option if the CSV is not found.

## Repository structure

A suggested project structure is:

```text
TikTok-Claim-Classification/
│
├── TikTok_Project_Lab.ipynb
├── tiktok_dataset.csv
└── README.md
```

The dataset is not included in this README. Make sure you have permission to use and share the dataset before adding it to a public repository.

## Reproducibility notes

- Run the notebook from the first cell through the final cell so that preprocessing, feature creation, model fitting, and evaluation occur in the correct order.
- The Random Forest and XGBoost grid searches use five-fold cross-validation and may take several minutes.
- The notebook uses `random_state=0` for the train/validation/test splits and model estimators where specified.
- Model results can vary with library versions, dataset versions, and execution environment.
- The saved outputs are historical notebook outputs; rerunning the notebook is necessary to verify the results on your own machine.
