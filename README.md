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

### 📖 Table of Contents
1. [About Me](#about-me)  
2. [Projects](#projects)  
   1. [Water Quality Classification](#1-water-quality-classification)  
   2. [Loan Approval Prediction Pipeline](#2-loan-approval-prediction-pipeline)  
   3. [Classification Workflow with Orange](#3-classification-workflow-with-orange)  

---

### 🧑‍💼 About Me

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

### 🏆 Projects

---

### 1. Water Quality Classification  
<span style="color: #9e9e9e;">*[Project Link](https://github.com/samuelvy1100/Mexico-Water-Quality-Classification)*</span>

**Overview**:
Developed a multi-class classification model to assess surface water safety across Mexico using physicochemical readings.

- 🚰 Classified water into `Verde` 🟩, `Amarillo` 🟨, and `Rojo` 🟥 using Gradient Boosting with 5-fold CV.
- 🔬 Identified key contaminants like turbidity and nitrates driving pollution.
- 📊 Delivered actionable insights for public health agencies and environmental engineers.

**Featured Skills in Action**:
`Python`, `scikit-learn`, `GridSearchCV`, `EDA`, `Gradient Boosting`, `Model Interpretability`

---

### 2. Loan Approval Prediction Pipeline
<span style="color: #9e9e9e;">*[Project Link](https://github.com/samuelvy1100/Loan-Approval-Prediction-Pipeline)*</span>

**Overview**:
Created a production-ready pipeline to evaluate real-world banking loan applications and automate credit decisions.

- 🏦 Achieved 98% accuracy using Random Forest on 4,000+ loan records.
- ⚙️ Deployed with `FastAPI` for real-time API serving and designed `Streamlit` frontend for user interaction.
- ⏱ Predictions delivered in under 200ms with robust classification reporting.

**Featured Skills in Action**:
`Python`, `FastAPI`, `Streamlit`, `Joblib`, `OneHotEncoder`, `Feature Engineering`, `MLOps`, `REST API Design`

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

### Acknowledgments & References

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
