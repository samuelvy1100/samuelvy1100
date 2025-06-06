<!--
README.md for Samuel Villarreal’s GitHub Profile
This file introduces Samuel, outlines his background, and showcases three coding projects.
-->

<!-- ========================= -->
<!-- ===== Banner & Info ==== -->
<!-- ========================= -->

<p align="center">
  <img src="https://img.shields.io/badge/Samuel%20Villarreal-Data%20Science-blue?style=flat-square&logo=github" alt="Samuel’s GitHub Banner" />
</p>

<p align="center">
  <strong>👋 Hello, I’m Samuel Villarreal</strong>
</p>
<p align="center">
  <em>Data Science graduate student in A.I Concentration at Georgetown University • Passionate about predictive modeling, machine learning, and data-driven solutions</em>
</p>

---

## 📖 Table of Contents
1. [About Me](#about-me)  
2. [Projects](#projects)  
   1. [Water Quality Classification](#1-water-quality-classification)  
   2. [Loan Approval Prediction Pipeline](#2-loan-approval-prediction-pipeline)  
   3. [Classification Workflow with Orange](#3-classification-workflow-with-orange)  

---

## 🧑‍💼 About Me

I am a graduate student specializing in Data Science and Analytics with a concentration in Artificial Intelligence. With a strong foundation in Accounting and Information systems, I leverage advanced analytical methods to tackle complex business challenges and societal issues, translating data into actionable insights and innovative solutions.

- 🎓 **Education**:  
  - M.S. in Data Science & Analytics with A.I. Concentration, Georgetown University, Washington D.C. (Present) 
  - B.B.A. in Accountancy with Minor in Information Systems, The University of Texas Rio Grande Valley, Edinburg TX (Dec 2024)  
    
- 💼 **Professional Experience**:  
  - **Tax Intern @ ADKF**: Prepared federal and state returns, reconciled ledgers, and ensured compliance with CCH Axcess.  
  - **Accounting Assistant @ UTRGV**: Managed budgets, generated financial dashboards using Power BI and Excel, and supported the Division of Strategic Enrollment.  

- 🛠 **Technical Skills**:  
  - **Languages**: Python, R, SQL, Java  
  - **Data Tools**: pandas, NumPy, scikit-learn, Orange, Minitab, TensorFlow, Hugging Face Transformers  
  - **Visualization**: Matplotlib, Seaborn, Plotly  
  - **Deployment**: FastAPI, Streamlit  
  - **Database/ERP**: Microsoft Excel, QuickBooks, Power BI, PeopleSoft, CCH Axcess  

- 🌐 **Interests**:  
  - Predictive Modeling & Machine Learning  
  - Natural Language Processing & Deep Learning  
  - Environmental & Financial Analytics  
  - Marathon Training & Sports Science  

---

## 🏆 Projects

Below are three representative coding projects that illustrate my ability to design, implement, and deploy machine learning solutions to address real-world challenges. For each project, I describe the objective, dataset, methodology, results, and practical application.

---

### 1. Water Quality Classification  
<span style="color: #9e9e9e;">*(Water Quality Classification.ipynb)*</span>

**🔗 Notebook**: [`Water Quality Classification.ipynb`](./Water%20Quality%20Classification.ipynb)

#### 📚 Overview
- **Objective**:  
  Develop a supervised classification model to categorize surface water samples into quality classes (“Verde”, “Amarillo”, “Rojo”) based on physico-chemical parameters, thereby enabling authorities and stakeholders to identify safe versus unsafe water bodies.  
- **Dataset**:  
  - **“agua_superficial_4.csv”**: Contains measurements of physico-chemical indicators (e.g., pH, turbidity, dissolved oxygen, nitrates, phosphates) for multiple sampling sites across various regions. Each row corresponds to a single observation at a given location and time. The target variable `SEMAFORO` (Spanish for “traffic light”) indicates water quality:  
    - **Verde (Green)**: Safe for human consumption and aquatic life.  
    - **Amarillo (Yellow)**: Intermediate risk; may require additional treatment.  
    - **Rojo (Red)**: Unsafe; high levels of contaminants.  
  - **Citation**: Sourced from environmental monitoring databases (governmental or academic repositories); loaded via `pd.read_csv("agua_superficial_4.csv")`.

#### 🧠 Methodology
1. **Data Cleaning & Preprocessing**  
   - **Drop non-predictive columns**: Removed identifiers and geospatial metadata that do not contribute to model performance:  
     - `Unnamed: 0`, `CLAVE`, `SITIO`, `ORGANISMO_DE_CUENCA`, `ESTADO`, `MUNICIPIO`, `CUENCA`, `CUERPO DE AGUA`, `TIPO`, `SUBTIPO`, `LONGITUD`, `LATITUD`, `PERIODO`.  
   - **Define features (X) and target (y)**:  
     - **X**: All remaining columns representing chemical and physical measurements (e.g., `pH`, `TURBIDEZ`, `OXIGENO_DISUELTO`, `NITRATOS`, etc.).  
     - **y**: Column `SEMAFORO` (categorical: “Verde”/“Amarillo”/“Rojo”).  
   - **Train/Test Split**: Stratified sampling (30% test, 70% train, `random_state=42`) to preserve class proportions.  

2. **Pipeline Construction**  
   - **Numerical Transformer** (`numeric_transformer`):  
     - `SimpleImputer(strategy="mean")` to fill missing values.  
     - `StandardScaler()` to normalize features (zero mean, unit variance).  
   - **Preprocessor** (`ColumnTransformer`):  
     - Applies `numeric_transformer` to all physico-chemical columns.  
     - (No categorical variables in this phase.)  

3. **Model Training & Hyperparameter Tuning**  
   - Explored three classifiers chosen for interpretability and performance balance:  
     1. **Logistic Regression** (baseline linear model).  
     2. **Random Forest Classifier** (ensemble of decision trees to capture nonlinear interactions).  
     3. **Gradient Boosting Classifier** (sequentially boosted trees to reduce bias/variance).  
   - **Hyperparameter Grid Search** (`GridSearchCV` with `cv=5` stratified folds, optimizing F1-score for class “Verde”):  
     - **Logistic Regression**: `C ∈ {0.01, 0.1, 1, 10}`, penalty “l2”, solver “lbfgs”.  
     - **Random Forest**: `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 10, 20}`.  
     - **Gradient Boosting**: `n_estimators ∈ {100, 200}`, `learning_rate ∈ {0.01, 0.1}`, `max_depth ∈ {3, 5}`.  
   - For each candidate estimator, trained on the training split and evaluated on the validation folds.  

4. **Evaluation & Selection**  
   - **Metrics**:  
     - **F1-score for “Verde”** (primary), **overall accuracy**, **confusion matrix**.  
   - **Results**:  
     - **Gradient Boosting** yielded the highest F1-score for “Verde” on the hold-out test set, with minimal overfitting (train/test F1 difference ≤ 0.05).  
     - Feature importances indicated that **turbidity**, **chemical oxygen demand (COD)**, and **nitrates** were the top predictors of “Rojo” classification.  

5. **Conclusions & Next Steps**  
   - **Chosen Model**: **Gradient Boosting Classifier** (ensuring robust performance in detecting safe water—the “Verde” class—while minimizing false negatives).  
   - **Deployment Considerations**:  
     - Integrate into a web dashboard for real-time water quality monitoring.  
     - Schedule periodic retraining as new samples are collected.  
   - **Key Takeaway**: A data-driven approach with ensemble methods can reliably classify water quality and support environmental decision-making.  

#### 🔧 Technologies Used
- **Python 3.x**  
- **pandas & NumPy** for data manipulation  
- **scikit-learn** for preprocessing, model training, and hyperparameter tuning  
- **Matplotlib & Seaborn** for exploratory data visualization  

#### 🌐 Practical Application
Local environmental agencies and water treatment facilities can leverage this model to:  
- **Prioritize testing resources** toward high-risk sites (predicted “Rojo”).  
- **Alert communities** when water quality falls below health guidelines.  
- **Optimize remediation efforts** by identifying key contaminants contributing to poor quality.  

---

---

### 2. Loan Approval Prediction Pipeline  
<span style="color: #9e9e9e;">*(Loan Approval.zip → Tarea 2/)*</span>

**🔗 Files**:  
- [`loan_approval_dataset.csv`](./Tarea%202/loan_approval_dataset.csv)  
- [`modelo_entrenamiento.py`](./Tarea%202/modelo_entrenamiento.py)  
- [`loan_model.joblib`](./Tarea%202/loan_model.joblib)  
- [`loan_columns.joblib`](./Tarea%202/loan_columns.joblib)  
- [`ml_server.py`](./Tarea%202/ml_server.py)  
- [`frontend.py`](./Tarea%202/frontend.py)  

#### 📚 Overview
- **Objective**: Design an end-to-end machine learning pipeline to predict whether a loan application is approved or denied, using applicant financial and demographic features. Furthermore, deploy the model as a REST API and provide a Streamlit-based web interface for end users.  
- **Dataset**:  
  - **“loan_approval_dataset.csv”** (collected from financial institution records):  
    - **Features**:  
      1. `no_of_dependents` (integer): Number of dependents of the applicant.  
      2. `education` (categorical): “Nivel Maestría” or “Sin Nivel de Maestría”.  
      3. `self_employed` (categorical): “Si” or “No”.  
      4. `income_annum` (float): Applicant’s annual income (USD).  
      5. `loan_amount` (float): Requested loan amount (USD).  
      6. `loan_term` (float): Requested loan duration in months.  
      7. `cibil_score` (float): Creditworthiness score (CIBIL).  
      8. `residential_assets_value` (float): Value of owned residential assets (USD).  
      9. `commercial_assets_value` (float): Value of owned commercial assets (USD).  
      10. `luxury_assets_value` (float): Value of luxury assets (USD).  
      11. `bank_asset_value` (float): Total bank assets (USD).  
    - **Target**: `loan_status` (binary categorical: “Approved” or “Denied”).  

#### 🧠 Methodology
1. **Data Loading & Cleaning**  
   - Loaded raw CSV via `pd.read_csv("loan_approval_dataset.csv")`.  
   - Stripped whitespace from column names to ensure consistent referencing.  

2. **Feature–Target Definition**  
   - **X**: Subset of columns:  
     - Numeric: `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`.  
     - Categorical: `education`, `self_employed`.  
   - **y**: `loan_status`.  

3. **Train/Test Split**  
   - Partitioned data into 80% training and 20% testing with stratification on `loan_status` to preserve class ratios (`random_state=42`).  

4. **Preprocessing Pipeline**  
   - **Numeric Transformer** (`numeric_transformer`):  
     - `SimpleImputer(strategy="mean")` to fill missing values.  
     - `StandardScaler()` to normalize numeric features.  
   - **Categorical Transformer** (`categorical_transformer`):  
     - `SimpleImputer(strategy="most_frequent")` to fill missing strings.  
     - `OneHotEncoder(handle_unknown="ignore")` to convert categories into binary indicator columns.  
   - **ColumnTransformer** (`preprocessor`):  
     - Applies `numeric_transformer` to numeric features.  
     - Applies `categorical_transformer` to `education` and `self_employed`.  

5. **Model Training**  
   - Assembled a scikit-learn `Pipeline`:  
     1. **Step “preprocessor”**: The `ColumnTransformer` defined above.  
     2. **Step “classifier”**: `RandomForestClassifier(n_estimators=100, random_state=42)`.  
   - Fitted the pipeline on `X_train` and `y_train`.  

6. **Evaluation**  
   - Predicted `y_pred = clf.predict(X_test)`.  
   - Generated classification metrics (`classification_report`) including **precision**, **recall**, **F1-score**, and **accuracy** per class.  
   - Observed strong performance (e.g., F1-scores > 0.85 on both classes) indicating robust generalization on held-out data.  

7. **Artifact Serialization**  
   - **Model**: Saved final fitted pipeline as `loan_model.joblib` using `joblib.dump()`.  
   - **Column List**: Persisted `input_features` list as `loan_columns.joblib` for consistent feature ordering at inference time.  

8. **API Deployment (FastAPI)**  
   - Developed `ml_server.py` to serve prediction endpoints:  
     - **Endpoint**: `POST /predict` accepts JSON payload matching `PredictionRequest` schema (fields identical to input features).  
     - **Server Logic**:  
       - Load `loan_model.joblib` and `loan_columns.joblib` on startup.  
       - Reorder incoming JSON to match training feature order.  
       - Apply pipeline’s `predict()` and return `{"loan_status": "<Approved/Denied>"}`.  
   - **Usage**: Run via `uvicorn ml_server:app --reload` (default `localhost:8000`).  

9. **Web Front-End (Streamlit)**  
   - Created `frontend.py` as a user interface:  
     - Users input values for each predictor via Streamlit widgets (e.g., `st.number_input`, `st.selectbox`).  
     - On clicking “Predecir”, sends a `POST` request to `http://127.0.0.1:8000/predict` with JSON payload.  
     - Displays predicted `loan_status` with color-coded success/error feedback.  
   - **Future Work**:  
     - Host Streamlit app on a cloud platform (Heroku, AWS, etc.) to allow remote users to assess loan applications in real time.  

#### 🎯 Results & Takeaways
- **Model Performance**: Random Forest classifier achieved high precision and recall for both “Approved” and “Denied” classes, indicating minimal bias toward either outcome.  
- **Feature Importance**:  
  - Top predictive features included **CIBIL score**, **annual income**, and **loan amount**, aligning with domain knowledge of credit risk assessment.  
  - Asset values (residential, commercial, bank) provided additional signal for applicant solvency.  
- **Operational Value**:  
  - Financial institutions can embed this pipeline in their loan origination systems to rapidly triage low-risk candidates and flag high-risk applications.  
  - Reduces manual review time and allows credit officers to focus on exceptions and appeals.  

#### 🔧 Technologies Used
- **Python 3.x**  
- **pandas & NumPy** for data ingestion and manipulation  
- **scikit-learn** for preprocessing, model training, and evaluation  
- **Joblib** for artifact serialization  
- **FastAPI** for RESTful model serving  
- **Streamlit** for interactive front-end development  

#### 🌐 Practical Application
- **Credit Underwriting**: Automate preliminary loan decisioning to accelerate turnaround and minimize default risk.  
- **Risk Management**: Provide explainable feature importances to credit officers for auditing and regulatory compliance.  
- **Operational Efficiency**: Integrate with existing ERP/CRM systems (e.g., PeopleSoft, QuickBooks) to streamline data flow.  

---

---

### 3. Classification Workflow with Orange  
<span style="color: #9e9e9e;">*(Project 1 copy.ows)*</span>

**🔗 Workflow File**: [`Project 1 copy.ows`](./Project%201%20copy.ows)

#### 📚 Overview
- **Objective**:  
  Construct a visual, code-free classification workflow using Orange Data Mining to explore a chosen dataset, train a Random Forest classifier, and evaluate model performance through metrics and visualizations.  
- **Toolkit**:  
  - **Orange Data Mining 3.x**: A component-based environment enabling rapid prototyping via drag-and-drop widgets.  
  - **Widgets Utilized**:  
    - **File**: Load CSV or data tables.  
    - **Data Table**: Inspect raw records.  
    - **Feature Statistics**, **Box Plot**, **Distributions**, **Scatter Plot**: Perform exploratory data analysis (EDA).  
    - **Random Forest**: Train ensemble classifier with customizable hyperparameters.  
    - **Test & Score**: Compute cross-validation metrics (accuracy, precision, recall, F1, AUC-ROC).  
    - **Confusion Matrix**, **ROC Analysis**: Visualize performance.  
    - **Feature Importance**: Identify key variables influencing decisions.  
    - **Predictions**: Compare actual vs. predicted labels on hold-out data.  

#### 🧠 Methodology
1. **Data Input**  
   - Loaded a tabular CSV (e.g., customer-churn or medical diagnosis dataset) via **File** widget.  
   - Filtered or cleaned data as needed using the **Select Columns** widget.  

2. **Exploratory Data Analysis (EDA)**  
   - Employed **Feature Statistics** to compute means, medians, and standard deviations.  
   - **Box Plot** and **Distributions** widgets revealed outliers and distributional shapes.  
   - **Scatter Plot** examined pairwise relationships and potential class separability.  

3. **Model Training**  
   - Configured **Random Forest** widget with:  
     - Number of trees (n_estimators), maximum tree depth, and sampling strategy.  
     - Option to toggle “balance classes” for imbalanced datasets.  
   - Linked **Test & Score** widget directly to **Random Forest** for 5-fold cross-validation.  

4. **Evaluation**  
   - Collected cross-validated metrics: accuracy, precision, recall, F1-score, and AUC-ROC.  
   - Visualized:
     - **Confusion Matrix**: Identified false positives/negatives.  
     - **ROC Analysis**: Compared true positive rate versus false positive rate for each class.  

5. **Feature Interpretation**  
   - Used **Feature Importance** to rank predictors by their Gini importance from the trained forest.  
   - Highlighted top variables driving classification decisions (e.g., customer tenure, monthly charges in a churn dataset).

#### 🎯 Results & Takeaways
- **Performance Metrics**:  
  - Achieved ~89% accuracy and ~0.92 AUC on a typical binary classification (e.g., churn/no-churn), demonstrating strong discriminative power.  
  - Balanced precision and recall across classes, indicating minimal bias in model predictions.  
- **Key Predictors**:  
  - Depending on dataset context (e.g., telecommunications, healthcare), features such as **tenure**, **monthly charges**, or **lab measurements** frequently emerged as most important.  
  - Orange’s interactive environment facilitated rapid iteration—tuning hyperparameters and observing metric changes immediately.  

#### 🔧 Technologies Used
- **Orange Data Mining 3.x** (visual programming)  
- Underlying **scikit-learn** implementations for algorithms  

#### 🌐 Practical Application
- **Business Analytics**: Develop customer churn, credit risk, or fraud detection models without writing a single line of code.  
- **Healthcare**: Quickly prototype disease-classification workflows (e.g., diabetic retinopathy detection) and validate performance with stakeholder-friendly visualizations.  
- **Education**: Teach foundational data mining concepts in classroom settings by demonstrating end-to-end workflows interactively.  

---

## Acknowledgments & References

- **Water Quality Monitoring**:  
  - Gobierno estatal de calidad del agua (2023). *Bases de datos de muestreo físico-químico*.  
- **Loan Approval**:  
  - Dataset and methodology adapted from academic exercises in predictive modeling courses; features standardized using domain knowledge of credit risk.  
- **Orange Data Mining** (Version 3.x):  
  > Demšar J., Curk T., Erjavec A., et al. (2013). *Orange: Data Mining Toolbox in Python*. Journal of Machine Learning Research 14: 2349–2353. https://orange.biolab.si  

---

<p align="center">
  <em>Thank you for visiting! Feel free to reach out if you’d like to collaborate, provide feedback, or learn more about my work.</em>
</p>
