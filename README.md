# MedPredict-ML
### An End-to-End Supervised Machine Learning System for Healthcare Analytics
#### *Implementing Every Concept from Andrew Ng's Course*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange?logo=numpy)](https://numpy.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

---

## Overview

**MedPredict-ML** is an implementation of every supervised machine learning algorithm covered in [Andrew Ng's Supervised Machine Learning course](https://www.coursera.org/learn/machine-learning), applied to **two real-world healthcare datasets**:

| Task | Dataset | Target |
|------|---------|--------|
| **Regression** | Medical Insurance Cost ([Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)) | Predict annual insurance charges (USD) |
| **Classification** | Pima Indians Diabetes ([UCI/Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)) | Predict diabetes risk (binary) |

Every algorithm, mathematical formula, and concept is implemented from scratch using only NumPy, then validated against scikit-learn. No black-boxes — every line of code corresponds directly to the mathematical equations in the course.

---

## Project Architecture

```
medpredict-ml/
│
├── src/
│   ├── algorithms.py          # All from-scratch implementations
│   │     ├── LinearRegression            (GD, regularised, normal eq.)
│   │     ├── LogisticRegression          (sigmoid, log-loss, regularised)
│   │     ├── FeatureScaler               (3 methods)
│   │     ├── PolynomialFeatureEngineer   (polynomial + interactions)
│   │     ├── VectorizationBenchmark      (speed comparison)
│   │     ├── sigmoid()
│   │     ├── check_convergence()
│   │     └── generate_learning_curves()
│   │
│   ├── preprocessing.py       # Data loading, splitting, EDA
│   └── visualization.py       # All plots (cost history, boundaries, etc.)
│
├── data/
│   ├── insurance.csv          # download from Kaggle
│   ├── diabetes.csv           # download from Kaggle
│   └── download_data.py       # Auto-download or generate synthetic data
│
├── results/                   
│
├── main_analysis.py           # 📌 MAIN FILE 
├── requirements.txt
└── README.md
```


## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/peter-adepoju/medpredict-ml.git
cd medpredict-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the datasets (choose one option)
#    Option A: Kaggle API
kaggle datasets download -d mirichoi0218/insurance -p data/ --unzip
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data/ --unzip

#    Option B: Synthetic fallback (auto-generated if CSVs not found)
python data/download_data.py

# 4. Run the full analysis
python main_analysis.py
```

## Acknowledgements

- **Course**: [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning) — Andrew Ng, DeepLearning.AI / Coursera
- **Datasets**: UCI Machine Learning Repository & Kaggle
- **Libraries**: NumPy, Pandas, Matplotlib, scikit-learn

---

## License

MIT License — see [LICENSE](LICENSE) for details.
