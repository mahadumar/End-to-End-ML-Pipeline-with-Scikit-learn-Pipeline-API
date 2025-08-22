Telco Customer Churn Prediction
===============================

Objective
---------

Build an end-to-end machine learning pipeline to predict customer churn for a telecom company, enabling proactive customer retention strategies and reducing customer acquisition costs.

Methodology
-----------

### 1\. Data Preprocessing

*   **Missing Values Handling**: Identified and removed 11 records with missing values in the TotalCharges column
    
*   **Data Type Conversion**: Converted TotalCharges from object to numeric format
    
*   **Categorical Encoding**: Applied one-hot encoding to all categorical variables with drop='first' to avoid multicollinearity
    
*   **Feature Scaling**: Standardized numerical features (tenure, MonthlyCharges, TotalCharges) using StandardScaler
    
*   **Target Variable Transformation**: Converted Churn values from 'Yes'/'No' to binary 1/0
    
*   **Train-Test Split**: 80-20 stratified split to maintain class distribution
    

### 2\. Model Training

*   **Algorithm Selection**: Implemented both Logistic Regression and Random Forest classifiers
    
*   **Pipeline Architecture**: Built Scikit-learn pipelines integrating preprocessing and modeling steps
    
*   **Logistic Regression**: Used L2 regularization with liblinear and lbfgs solvers
    
*   **Random Forest**: Ensemble method with multiple decision trees for improved accuracy
    

### 3\. Hyperparameter Tuning

*   **GridSearchCV**: Implemented exhaustive search over specified parameter values
    
*   **Cross-Validation**: 5-fold cross-validation for robust parameter estimation
    
*   **Logistic Regression Parameters**:
    
    *   C: \[0.1, 1, 10\] (regularization strength)
        
    *   solver: \['liblinear', 'lbfgs'\]
        
*   **Random Forest Parameters**:
    
    *   n\_estimators: \[100, 200\] (number of trees)
        
    *   max\_depth: \[None, 10, 20\] (tree depth)
        
    *   min\_samples\_split: \[2, 5\] (minimum samples to split)
        

### 4\. Evaluation Metrics

*   **Accuracy**: Overall prediction correctness
    
*   **Precision**: Proportion of true positives among predicted positives
    
*   **Recall**: Proportion of actual positives correctly identified
    
*   **F1-Score**: Harmonic mean of precision and recall
    
*   **ROC-AUC**: Area under Receiver Operating Characteristic curve
    
*   **Confusion Matrix**: Visual representation of prediction results
    

### 5\. Model Export

*   **Joblib Serialization**: Saved complete pipeline including preprocessing steps
    
*   **Production Ready**: Pipeline handles new data with same preprocessing
    
*   **Reusability**: Single file deployment for easy integration
    

Key Results
-----------

### Performance Comparison

**ModelAccuracyPrecisionRecallF1-ScoreROC-AUC**Logistic Regression0.810.660.530.590.84Random Forest0.800.700.500.58**0.85**

### Feature Importance

*   **Top Predictive Features**:
    
    1.  Tenure (customer loyalty duration)
        
    2.  Monthly Charges (service cost)
        
    3.  Contract Type (month-to-month vs long-term)
        
    4.  Payment Method (electronic check vs others)
        
    5.  Internet Service Type
        

### Business Insights

*   Customers with shorter tenure (<12 months) have 3x higher churn risk
    
*   Month-to-month contracts show 45% higher churn rate than annual contracts
    
*   Electronic check users are 2.5x more likely to churn than automatic payment users
    
*   Fiber optic internet customers have 30% higher churn rate than DSL users
    

Files
-----

*   churn\_prediction.ipynb: Complete Jupyter notebook with detailed analysis and code
    
*   churn\_prediction\_pipeline.pkl: Exported production-ready pipeline
    
*   dataset/: Directory containing the Telco Customer Churn dataset
    

Installation & Usage
--------------------

bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Install required packages  pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter  # Run Jupyter notebook  jupyter notebook churn_prediction.ipynb  # Load model in production  import joblib  pipeline = joblib.load('churn_prediction_pipeline.pkl')  predictions = pipeline.predict(new_data)   `

Future Enhancements
-------------------

*   Implement class imbalance techniques (SMOTE, class weights)
    
*   Add advanced models (XGBoost, Gradient Boosting)
    
*   Create REST API for real-time predictions
    
*   Develop customer retention recommendation system
    
*   Implement model monitoring and drift detection
    

Technical Stack
---------------

*   **Programming Language**: Python 3.8+
    
*   **ML Framework**: Scikit-learn
    
*   **Data Processing**: Pandas, NumPy
    
*   **Visualization**: Matplotlib, Seaborn
    
*   **Serialization**: Joblib
    
*   **Development**: Jupyter Notebook
    

This pipeline provides a robust foundation for customer churn prediction and can be easily extended for production deployment and real-time inference capabilities.
