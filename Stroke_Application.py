# Author: Tabea Attig

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import classification_report, auc, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, matthews_corrcoef, r2_score, precision_score, f1_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import io
import base64

# Load data from uploaded files
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('tsv'):
            df = pd.read_csv(uploaded_file, sep = "\t")
        elif uploaded_file.name.endswith('xml'):
            df = pd.read_xml(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df

# Create a download link for the dataframe
def download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'

# Function to save a plot to a BytesIO object and provide a download button
def plot_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf


# Main function
def main():
    st.set_page_config(page_title='AutoML App', layout='wide', initial_sidebar_state='expanded')
    st.title('🧠 AUTOMATED MACHINE LEARNING APP')

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv", "xml"])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("### 🗂️ DATA OVERVIEW")
            st.write(df.head())
            
            # Column Selection for Dropping
            columns_to_drop = st.multiselect('Select Columns to Drop', df.columns.tolist())
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                st.success(f"Columns {columns_to_drop} dropped.")
                st.write(df.head())
            
            # Column types
            total_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
            
            # EDA
            st.markdown("### 🔍 EXPLORATORY DATA ANALYSIS")

            with st.expander('Data Details'):
                if st.checkbox('Show Summary Statistics'):
                    st.write(df.describe())

                if st.checkbox('Show Data Types'):
                    st.write(df.dtypes)
                
                # Show Shape of Data
                if st.checkbox('Show Data Shape'):
                    st.write(f"Data Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

                # Show Missing Values
                if st.checkbox('Show Missing Values'):
                    missing_values = df.isna().sum()
                    st.write(missing_values[missing_values > 0])

                # Show Data Info
                if st.checkbox('Show Data Info'):
                    buffer = io.StringIO() 
                    df.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(s)  # Display the info

                # Show Duplicated Rows
                if st.checkbox('Show Duplicated Rows'):
                    duplicated_rows = df[df.duplicated()]
                    st.write(duplicated_rows if len(duplicated_rows) > 0 else "No duplicated rows found.")
                

            # Create tabs for different plots
            tab1, tab2, tab3, tab4 = st.tabs(["Pairplot", "Correlation Heatmap", "Histograms", "Countplots" ])

            with tab1:
                # Pairplot for numeric data
                if st.checkbox('Show Pairplot (Numeric Data)'):
                    if len(numeric_cols) > 1:
                        pairplot_fig = sns.pairplot(df[numeric_cols])
                        st.pyplot(pairplot_fig)  
                        buf = plot_to_bytes(pairplot_fig) 
                        st.download_button(
                            label="⬇️ Download Pairplot",
                            data=buf,
                            file_name="pairplot.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("Not enough numeric columns for a pairplot.")

            with tab2:
                # Correlation heatmap for numeric data
                if st.checkbox('Show Correlation Heatmap (Numeric Data)'):
                    if len(numeric_cols) > 1:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                        st.pyplot(fig)  
                        buf = plot_to_bytes(fig) 
                        st.download_button(
                            label="⬇️ Download Correlation Heatmap",
                            data=buf,
                            file_name="correlation_heatmap.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("Not enough numeric columns to display a correlation heatmap.")

            with tab3:
                # Countplots for categorical data
                if st.checkbox('Show Countplots (Categorical Data)'):
                    if len(categorical_cols) > 0:
                        num_plots = len(categorical_cols)
                        cols = 2
                        rows = (num_plots + 1) // cols  

                        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
                        axes = axes.flatten()  

                        for i, col in enumerate(categorical_cols):
                            sns.countplot(data=df, x=col, ax=axes[i], palette="Set2")
                            axes[i].set_title(f'Countplot of {col}')
                            axes[i].tick_params(axis='x', rotation=45)

                            for p in axes[i].patches:
                                height = p.get_height()
                                axes[i].text(p.get_x() + p.get_width() / 2, height, int(height), 
                                            ha='center', va='bottom')

                            max_height = df[col].value_counts().max()
                            axes[i].set_ylim(0, max_height * 1.1)  

                        for i in range(num_plots, rows * cols):
                            fig.delaxes(axes[i])

                        plt.subplots_adjust(hspace=0.8, wspace=0.3)  
                        st.pyplot(fig)  
                        buf = plot_to_bytes(fig) 
                        st.download_button(
                            label="⬇️ Download Count Plots of Categorical Columns",
                            data=buf,
                            file_name="count_plots_categorical_cols.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("No categorical columns available for countplots.")

            with tab4:
                # Plot histograms for numerical columns
                if st.checkbox('Show Histograms (Numerical Data)'):
                    if len(numeric_cols) > 0:
                        num_plots = len(numeric_cols)
                        cols = 2
                        rows = (num_plots + 1) // cols

                        fig_hist, axes_hist = plt.subplots(rows, cols, figsize=(12, 4 * rows))
                        axes_hist = axes_hist.flatten()

                        for i, col in enumerate(numeric_cols):
                            sns.histplot(df[col], kde=True, bins=30, palette="Set2", ax=axes_hist[i])
                            axes_hist[i].set_title(f'Distribution of {col}')
                            axes_hist[i].set_xlabel(col)
                            axes_hist[i].set_ylabel('Frequency')

                        for i in range(num_plots, rows * cols):
                            fig_hist.delaxes(axes_hist[i])

                        plt.subplots_adjust(hspace=0.8, wspace=0.3)
                        st.pyplot(fig_hist)  
                        buf_hist = plot_to_bytes(fig_hist)  
                        st.download_button(
                            label="⬇️ Download Histograms of Numerical Columns",
                            data=buf_hist,
                            file_name="histograms_numerical_cols.png",
                            mime="image/png"
                        )


            # Preprocessing
            st.markdown("### ⚙️ PREPROCESSING")

            # Select target variable
            st.markdown("#### 🎯 Target Variable")
            target_variable = st.selectbox('Select Target Variable', df.columns)
            
            # Missing Values Handling
            st.markdown("#### 🛠️ Handle Missing Values")
            missing_strategy = st.selectbox('Select Missing Values Strategy', ['mean', 'median'])
            if st.checkbox('Handle Missing Values'):
                imputer = SimpleImputer(strategy=missing_strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                st.write(df.head())
            
            # Encoding
            st.markdown("#### 🧩 Encoding")
            st.write("**Label Encoding**: This technique converts categorical values into numerical values by assigning a unique integer to each category. It’s suitable for ordinal data where the order of categories matters.")
            st.write("**One-Hot Encoding**: This method converts categorical variables into a binary matrix, where each category is represented by a binary column. It’s useful for nominal data where categories don’t have an intrinsic order.")
            encoding_method = st.selectbox('Select Encoding Method', ['None', 'Label Encoding', 'One-Hot Encoding'])
            if encoding_method != 'None':
                if encoding_method == 'Label Encoding':
                    le = LabelEncoder()
                    for col in categorical_cols:
                        df[col] = le.fit_transform(df[col].astype(str))
                elif encoding_method == 'One-Hot Encoding':
                    df = pd.get_dummies(df, columns=categorical_cols)
                st.write("### Encoded Data")
                st.write(df.head())

            # Model Selection
            st.markdown("### 🔧 MODEL SELECTION")
            model_type = st.selectbox('Select Machine Learning Type', ['Classification', 'Regression'])

            model = None
            param_grid = None

            if model_type == 'Classification':
                model_choice = st.selectbox('Select Classification Model', ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'KNeighbors', 'Naive Bayes', 'SVC', 'XGBoost'])

                if model_choice == 'Logistic Regression':
                    st.write("""
                    **Description**: Logistic Regression is a widely used linear model for binary and multiclass classification problems. It predicts probabilities using a logistic function.
                    """)
                    model = LogisticRegression(max_iter=1000)
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'solver': ['liblinear', 'lbfgs'],
                        'class_weight': [None, 'balanced']
                    }
                    default_values = {'C': 1, 'solver': 'lbfgs', 'class_weight': None}

                elif model_choice == 'Random Forest':
                    st.write("""
                    **Description**: Random Forest is an ensemble learning method that uses multiple decision trees to improve classification accuracy and control overfitting.
                    """)
                    model = RandomForestClassifier()
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [5, 10, 20, None],
                        'class_weight': [None, 'balanced']
                    }
                    default_values = {'n_estimators': 100, 'max_depth': None, 'class_weight': None}

                elif model_choice == 'Gradient Boosting':
                    st.write("""
                    **Description**: Gradient Boosting is an ensemble technique that builds models sequentially to correct errors made by previous models, enhancing predictive performance.
                    """)
                    model = GradientBoostingClassifier()
                    param_grid = {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'n_estimators': [100, 200]
                    }
                    default_values = {'learning_rate': 0.1, 'n_estimators': 100}

                elif model_choice == 'KNeighbors':
                    st.write("""
                    **Description**: K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies data points based on the majority label of their nearest neighbors.
                    """)
                    model = KNeighborsClassifier()
                    param_grid = {
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance']
                    }
                    default_values = {'n_neighbors': 5, 'weights': 'uniform'}

                elif model_choice == 'Naive Bayes':
                    st.write("""
                    **Description**: Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features, suitable for text classification and large datasets.
                    """)
                    model = GaussianNB()
                    param_grid = None

                elif model_choice == 'SVC':
                    st.write("""
                    **Description**: Support Vector Classifier aims to find a hyperplane that best separates different classes in the feature space, effective for high-dimensional data.
                    """)
                    model = SVC()
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'class_weight': [None, 'balanced']
                    }
                    default_values = {'C': 1, 'kernel': 'rbf', 'class_weight': None}

                elif model_choice == 'XGBoost':
                    st.write("""
                    **Description**: XGBoost is an optimized gradient boosting library designed for speed and performance, widely used in machine learning competitions.
                    """)
                    model = xgb.XGBClassifier()
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 6, 9],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    default_values = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8}

            elif model_type == 'Regression':
                model_choice = st.selectbox('Select Regression Model', ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'SVR', 'ElasticNet', 'XGBoost'])

                if model_choice == 'Linear Regression':
                    st.write("""
                    **Description**: Linear Regression models the relationship between a dependent variable and one or more independent variables using a linear approach.
                    """)
                    model = LinearRegression()
                    param_grid = None

                elif model_choice == 'Ridge Regression':
                    st.write("""
                    **Description**: Ridge Regression applies L2 regularization to linear regression to prevent overfitting by adding a penalty proportional to the square of the coefficients.
                    """)
                    model = Ridge()
                    param_grid = {
                        'alpha': [0.1, 1, 10]
                    }
                    default_values = {'alpha': 1}

                elif model_choice == 'Lasso Regression':
                    st.write("""
                    **Description**: Lasso Regression applies L1 regularization to linear regression, encouraging sparsity by adding a penalty proportional to the absolute value of the coefficients.
                    """)
                    model = Lasso()
                    param_grid = {
                        'alpha': [0.1, 1, 10]
                    }
                    default_values = {'alpha': 1}

                elif model_choice == 'Random Forest':
                    st.write("""
                    **Description**: Random Forest is an ensemble learning method that combines multiple decision trees to improve regression accuracy and reduce overfitting.
                    """)
                    model = RandomForestRegressor()
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [5, 10, 20, None]
                    }
                    default_values = {'n_estimators': 100, 'max_depth': None}

                elif model_choice == 'SVR':
                    st.write("""
                    **Description**: Support Vector Regression uses the principles of Support Vector Machines to perform regression, finding a function that deviates minimally from the training data.
                    """)
                    model = SVR()
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf']
                    }
                    default_values = {'C': 1, 'kernel': 'rbf'}

                elif model_choice == 'ElasticNet':
                    st.write("""
                    **Description**: ElasticNet combines L1 and L2 regularization to linear regression, balancing the penalties to improve model performance and feature selection.
                    """)
                    model = ElasticNet()
                    param_grid = {
                        'alpha': [0.1, 1, 10],
                        'l1_ratio': [0.1, 0.5, 0.9]
                    }
                    default_values = {'alpha': 1, 'l1_ratio': 0.5}

                elif model_choice == 'XGBoost':
                    st.write("""
                    **Description**: XGBoost is an optimized gradient boosting library designed for speed and performance, widely used in machine learning competitions.
                    """)
                    model = xgb.XGBRegressor()
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 6, 9],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    default_values = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8}


            # Validation Strategy
            st.markdown("### 🧪 MODEL VALIDATION")

            if model_type == 'Classification':
                validation_strategy = st.selectbox('🔍 Choose Validation Strategy', ['Train-Test Split', 'Stratified K-Fold Cross-Validation'])
            else:
                validation_strategy = st.selectbox('🔍 Choose Validation Strategy', ['Train-Test Split', 'K-Fold Cross-Validation'])

            if validation_strategy == 'K-Fold Cross-Validation' or validation_strategy == 'Stratified K-Fold Cross-Validation':
                n_splits = st.slider('Select Number of Folds (K)', min_value=2, max_value=10, value=5)
                
                # Grid Search Option
                use_grid_search = st.checkbox('Use Grid Search for Hyperparameter Tuning')

        
                if use_grid_search:
                    # Select Scoring Method
                    scoring_options = {
                        'Accuracy (Classification)': 'accuracy',
                        'Balanced Accuracy (Classification)': 'balanced_accuracy',
                        'Precision (Classification)': 'precision',
                        'Recall (Classification)': 'recall',
                        'F1 (Classification)': 'f1',
                        'F1 Macro (Classification)': 'f1_macro',
                        'ROC AUC (Classification)': 'roc_auc',
                        'Neg Mean Squared Error (Regression)': 'neg_mean_squared_error',
                        'Neg Mean Absolute Error (Regression)': 'neg_mean_absolute_error'
                    }
                    selected_scoring = st.selectbox('Select Scoring Method', options=list(scoring_options.keys()), index=0)
                    scoring = scoring_options[selected_scoring]

                    if param_grid is not None:
                        st.markdown("#### 🔍 Hyperparameter Tuning")
                        for param, values in param_grid.items():
                            default_value = default_values.get(param, values[0])
                            st.multiselect(f'**{param}**', options=values, default=[default_value])

                            if param == 'C':
                                st.write(f"**Explanation**: `{param}` controls the inverse of regularization strength. Smaller values specify stronger regularization.")
                            elif param == 'solver':
                                st.write(f"**Explanation**: `{param}` determines the algorithm used to optimize the model. Options include 'liblinear' and 'lbfgs'.")
                            elif param == 'class_weight':
                                st.write(f"**Explanation**: `{param}` adjusts weights inversely proportional to class frequencies in the training set.")
                            elif param == 'n_estimators':
                                st.write(f"**Explanation**: `{param}` specifies the number of trees in the forest.")
                            elif param == 'max_depth':
                                st.write(f"**Explanation**: `{param}` controls the maximum depth of the trees.")
                            elif param == 'learning_rate':
                                st.write(f"**Explanation**: `{param}` controls the step size at each iteration while moving towards a minimum.")
                            elif param == 'kernel':
                                st.write(f"**Explanation**: `{param}` specifies the type of kernel to be used in the SVM model. Options include 'linear' and 'rbf'.")
                            elif param == 'alpha':
                                st.write(f"**Explanation**: `{param}` is a regularization parameter for Ridge and Lasso regression, controlling the strength of the penalty.")
                            elif param == 'l1_ratio':
                                st.write(f"**Explanation**: `{param}` is the mixing parameter between L1 and L2 regularization in ElasticNet.")
                            elif param == 'subsample':
                                st.write(f"**Explanation**: `{param}` is the fraction of samples used to fit each base learner.")
            else:
                # Hide grid search options when Train-Test Split is selected
                st.write("Grid search options are not available with Train-Test Split.")

            # Advanced methods
            st.markdown("### 🚀 ADVANCED METHODS")

            # Standardization Option
            st.markdown("#### 🔄 Data Standardization")
            st.write("**Standardization**: Standardization is a preprocessing technique that involves scaling features. It is useful when the features in your dataset have different units or scales.")

            scaler_method = st.selectbox('Select Standardization Method', ['None', 'StandardScaler', 'MinMaxScaler'])
            
            # PCA Option
            st.markdown("#### 📉 Dimensionality Reduction")

            # Additional PCA information
            st.write("**PCA**: PCA is useful when you have high-dimensional data and you want to reduce the number of features while retaining the most important information.")

            apply_pca = st.checkbox('Apply PCA', value=False)
            if apply_pca:
                n_components = st.slider('Number of PCA Components', 1, min(len(total_cols), 10))

            st.markdown("#### ⚖️ Resampling for Classification")
            st.write("No resampling available for regression.")
        
            resampling_strategy = st.selectbox('Choose Resampling Technique', ['None', 'SMOTE', 'ADASYN', 'RandomUnderSampler'])
            
            if resampling_strategy == "SMOTE":
                st.info("**Synthetic Minority Over-sampling Technique (SMOTE)** is used to address class imbalance by creating synthetic samples for the minority class.")
            elif resampling_strategy == "ADASYN":
                st.info("**Adaptive Synthetic Sampling (ADASYN)** is an extension of SMOTE. It generates synthetic samples to balance the class distribution, with a focus on generating samples for minority class instances that are difficult to classify. ADASYN adjusts the distribution of synthetic samples based on the density of the minority class instances.")
            elif resampling_strategy == "RandomUnderSampler":    
                st.info("**Random Under-Sampling (RandomUnderSampler)** reduces the number of samples in the majority class to balance the class distribution. This technique randomly selects a subset of the majority class samples to match the number of minority class samples.")

            
            # Model Training
            st.markdown("### 🏋️‍♂️ TRAIN MODEL")
            if st.button('Train Model'):
                X = df.drop(columns=[target_variable])
                y = df[target_variable]


                if model_type in ['Classification', 'Regression']:
                    if validation_strategy == 'Train-Test Split':
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Apply Data Standardization
                        if scaler_method != 'None':
                            if scaler_method == 'StandardScaler':
                                scaler = StandardScaler()
                            elif scaler_method == 'MinMaxScaler':
                                scaler = MinMaxScaler()

                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)

                        # Apply PCA
                        if apply_pca:
                            pca = PCA(n_components=n_components)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)
                        
                        # Apply resampling
                        if model_type == 'Classification':
                            if resampling_strategy != 'None':
                                if resampling_strategy == 'SMOTE':
                                    sampler = SMOTE()
                                elif resampling_strategy == 'ADASYN':
                                    sampler = ADASYN()
                                elif resampling_strategy == 'RandomUnderSampler':
                                    sampler = RandomUnderSampler()

                                X_train, y_train = sampler.fit_resample(X_train, y_train)
                        
            
                        # Train the model
                        model.fit(X_train, y_train)
                        
                        # Save the trained model and data
                        st.session_state.model = model
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.validation_strategy = validation_strategy
                        
                        st.success('Model trained successfully! You can now evaluate it.')

                        
                    elif validation_strategy == 'K-Fold Cross-Validation' or validation_strategy == 'Stratified K-Fold Cross-Validation':
                        if model_type == 'Classification':
                            # Use StratifiedKFold to maintain class distribution in each fold
                            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        else:
                            # Use KFold for regression
                            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

                        # Initialize lists to store metrics across folds
                        roc_auc_scores = []
                        precision_scores_macro = []
                        recall_scores_macro = []
                        f1_scores_macro = []
                        precision_scores_weighted = []
                        recall_scores_weighted = []
                        f1_scores_weighted = []
                        mcc_scores = []
                        r2_scores = []
                        mse_scores = []
                        mae_scores = []
                        accuracy_scores = []

                        # Initialize lists to store ROC curve data for plotting
                        mean_fpr = np.linspace(0, 1, 100)
                        tprs = []
                        mean_recall = np.linspace(0, 1, 100)
                        precisions = []

                        # Perform K-Fold CV
                        for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        
                            # Apply Data Standardization
                            if scaler_method != 'None':
                                if scaler_method == 'StandardScaler':
                                    scaler = StandardScaler()
                                elif scaler_method == 'MinMaxScaler':
                                    scaler = MinMaxScaler()

                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                                
                            # Apply PCA
                            if apply_pca:
                                pca = PCA(n_components=n_components)
                                X_train = pca.fit_transform(X_train)
                                X_test = pca.transform(X_test)
                            
                            # Apply resampling
                            if model_type == 'Classification':
                                if resampling_strategy != 'None':
                                    if resampling_strategy == 'SMOTE':
                                        sampler = SMOTE()
                                    elif resampling_strategy == 'ADASYN':
                                        sampler = ADASYN()
                                    elif resampling_strategy == 'RandomUnderSampler':
                                        sampler = RandomUnderSampler()

                                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                            
                            if use_grid_search and param_grid:
                                #st.write("Performing Grid Search...")
                                grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring)
                                grid_search.fit(X_train, y_train)
                                model = grid_search.best_estimator_
                                #st.write(f"Best Hyperparameters: {grid_search.best_params_}")
                        
                            # Train the model on this fold
                            model.fit(X_train, y_train)

                            # Make predictions
                            y_pred = model.predict(X_test)
                            y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                            # If it's a classification model
                            if model_type == 'Classification':
                                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                                roc_auc = auc(fpr, tpr)
                                roc_auc_scores.append(roc_auc)
                                accuracy_scores.append(accuracy_score(y_test, y_pred))
                                precision_scores_macro.append(precision_score(y_test, y_pred, average='macro'))
                                recall_scores_macro.append(recall_score(y_test, y_pred, average='macro'))
                                f1_scores_macro.append(f1_score(y_test, y_pred, average='macro'))
                                precision_scores_weighted.append(precision_score(y_test, y_pred, average='weighted'))
                                recall_scores_weighted.append(recall_score(y_test, y_pred, average='weighted'))
                                f1_scores_weighted.append(f1_score(y_test, y_pred, average='weighted'))
                                mcc_scores.append(matthews_corrcoef(y_test, y_pred))
                                    
                                # ROC curve
                                tpr_interpolated = np.interp(mean_fpr, fpr, tpr)
                                tpr_interpolated[0] = 0.0
                                tprs.append(tpr_interpolated)

                                # Precision-Recall curve
                                precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
                                precision_interpolated = np.interp(mean_recall, recall[::-1], precision[::-1])
                                precisions.append(precision_interpolated)

                            elif model_type == 'Regression':
                                mse_scores.append(mean_squared_error(y_test, y_pred))
                                r2_scores.append(r2_score(y_test, y_pred))
                                mae_scores.append(mean_absolute_error(y_test, y_pred))

                        # Save trained model
                        st.session_state.model = model
                        st.session_state.validation_strategy = validation_strategy
                        st.session_state.kf_scores = {
                            "accuracy": accuracy_scores,
                            "roc_auc": roc_auc_scores,
                            "macro_precision": precision_scores_macro,
                            "macro_recall": recall_scores_macro,
                            "macro_f1": f1_scores_macro,
                            "weighted_precision": precision_scores_weighted,
                            "weighted_recall": recall_scores_weighted,
                            "weighted_f1": f1_scores_weighted,
                            "mcc": mcc_scores,
                            "r2": r2_scores,
                            "mse": mse_scores,
                            "mae": mae_scores
                        }
                        st.session_state.tprs = tprs
                        st.session_state.precisions = precisions
                        st.success('Model trained successfully with K-Fold Cross-Validation! You can now evaluate it.')

            # Evaluation
            if 'model' in st.session_state and model_type in ['Classification', 'Regression']:
                if validation_strategy == 'Train-Test Split':
                    model = st.session_state.model
                    X_test, y_test = st.session_state.X_test, st.session_state.y_test
                    y_pred = model.predict(X_test)

                    if model_type == 'Classification':
                        st.markdown("## 📊 Results for Classification")
                        st.write("### Classification Report")
                        st.text(classification_report(y_test, y_pred))

                        # Matthews Correlation Coefficient
                        st.write("### Matthews Correlation Coefficient")
                        mcc = matthews_corrcoef(y_test, y_pred)
                        st.write(f"Matthews Correlation Coefficient: {mcc:.4f}")
                        
                        # Create tabs for different plots
                        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])

                        with tab1:
                            if st.checkbox('Show Confusion Matrix'):
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)
                                buf = plot_to_bytes(fig)
                                st.download_button(
                                    label="⬇️ Download Confusion Matrix",
                                    data=buf,
                                    file_name="confusion_matrix_traintest.png",
                                    mime="image/png"
                                )

                        with tab2:
                            if st.checkbox('Show ROC Curve (supported only for binary labels)'):
                                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                                fig, ax = plt.subplots()
                                ax.plot(fpr, tpr, marker='.')
                                ax.plot([0, 1], [0, 1], linestyle='--', label='random', color='red')
                                ax.set_xlabel('False Positive Rate')
                                ax.set_ylabel('True Positive Rate')
                                ax.set_title('ROC Curve')
                                st.pyplot(fig)
                                buf = plot_to_bytes(fig)
                                
                                st.download_button(
                                    label="⬇️ Download ROC Curve",
                                    data=buf,
                                    file_name="roc_curve_traintest.png",
                                    mime="image/png"
                                )
                        
                        with tab3:
                            if st.checkbox('Show Precision-Recall Curve (supported only for binary labels)'):
                                precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
                                fig, ax = plt.subplots()
                                ax.plot(recall, precision, marker='.')
                                ax.set_xlabel('Recall')
                                ax.set_ylabel('Precision')
                                ax.set_title('Precision-Recall Curve')
                                st.pyplot(fig)
                                buf = plot_to_bytes(fig)

                                st.download_button(
                                    label="⬇️ Download Precision-Recall Curve",
                                    data=buf,
                                    file_name="precision_recall_curve_traintest.png",
                                    mime="image/png"
                                )
                    
                    elif model_type == 'Regression':
                        st.markdown("## 📊 Results for Regression")

                        y_pred = model.predict(X_test)

                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)

                        st.write(f"**Mean Squared Error:** {mse:.4f}")
                        st.write(f"**R^2 Score:** {r2:.4f}")
                        st.write(f"**Mean Absolute Error:** {mae:.4f}")

                        # Best Overall Plot: Predicted vs Actual Plot
                        st.markdown("### 📈 Best Overall Model Performance Plot: Predicted vs. Actual Plot")
                        st.write("Why: This is a great all-around plot to visually inspect how well the model’s predictions align with actual values. "
                                 "If the points fall along the diagonal line, the model is performing well.")
                        
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title('Predicted vs Actual')
                        st.pyplot(fig)

                        # Download option
                        buf = plot_to_bytes(fig)
                        st.download_button(
                            label="⬇️ Download Predicted vs. Actual Plot",
                            data=buf,
                            file_name="predicted_vs_actual_traintest.png",
                            mime="image/png"
                        )

                elif validation_strategy == 'K-Fold Cross-Validation' or validation_strategy == 'Stratified K-Fold Cross-Validation':
                    if model_type == 'Classification':
                        st.markdown("## 📊 K-Fold Cross-Validation Results for Classification")

                        # Calculate means and standard deviations
                        mean_metrics = {key: np.mean(val) for key, val in st.session_state.kf_scores.items()}
                        std_metrics = {key: np.std(val) for key, val in st.session_state.kf_scores.items()}

                        # Organizing metrics into two rows for better layout
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col6, col7, col8, col9 = st.columns(4)

                        # First row of metrics
                        with col1:
                            st.metric(label="Average Accuracy", value=f"{mean_metrics['accuracy']:.4f}", delta=f"{std_metrics['accuracy']:.4f}")
                        with col2:
                            st.metric(label="Average ROC-AUC", value=f"{mean_metrics['roc_auc']:.4f}", delta=f"{std_metrics['roc_auc']:.4f}")
                        with col3:
                            st.metric(label="Average MCC", value=f"{mean_metrics['mcc']:.4f}", delta=f"{std_metrics['mcc']:.4f}")
                        with col4:
                            st.metric(label="Average Precision (weighted)", value=f"{mean_metrics['weighted_precision']:.4f}", delta=f"{std_metrics['weighted_precision']:.4f}")
                        with col5:
                            st.metric(label="Average Recall (weighted)", value=f"{mean_metrics['weighted_recall']:.4f}", delta=f"{std_metrics['weighted_recall']:.4f}")

                        # Second row of metrics
                        with col6:
                            st.metric(label="Average F1-Score (weighted)", value=f"{mean_metrics['weighted_f1']:.4f}", delta=f"{std_metrics['weighted_f1']:.4f}")
                        with col7:
                            st.metric(label="Average Precision (macro)", value=f"{mean_metrics['macro_precision']:.4f}", delta=f"{std_metrics['macro_precision']:.4f}")
                        with col8:
                            st.metric(label="Average Recall (macro)", value=f"{mean_metrics['macro_recall']:.4f}", delta=f"{std_metrics['macro_recall']:.4f}")
                        with col9:
                            st.metric(label="Average F1-Score (macro)", value=f"{mean_metrics['macro_f1']:.4f}", delta=f"{std_metrics['macro_f1']:.4f}")

                        # Create tabs for different plots
                        tab1, tab2 = st.tabs(["ROC Curve", "Precision-Recall Curve"])
                        
                        with tab1:
                            # Plot ROC-AUC Scores
                            st.markdown("### 📈 ROC Curve")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for fold, tpr in enumerate(st.session_state.tprs, 1):
                                fpr = np.linspace(0, 1, 100)
                                ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold}')
                            mean_tpr = np.mean(st.session_state.tprs, axis=0)
                            mean_auc = auc(mean_fpr, mean_tpr)
                            ax.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.4f})', lw=2, alpha=0.8)
                            ax.plot([0, 1], [0, 1], linestyle='--', label='Random', color='red')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve for K-Fold Cross-Validation')
                            ax.legend(loc='lower right')
                            st.pyplot(fig)

                            buf = plot_to_bytes(fig)
                            st.download_button(
                                label="⬇️ Download ROC Curves",
                                data=buf,
                                file_name="roc_curves_kfold.png",
                                mime="image/png"
                            )   

                        with tab2: 
                            # Plot Precision-Recall Curves
                            st.markdown("### 📈 Precision-Recall Curve")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            for fold, precision in enumerate(st.session_state.precisions, 1):
                                recall = np.linspace(0, 1, 100)
                                ax.plot(recall, precision, alpha=0.3, label=f'Fold {fold}')
                            mean_precision = np.mean(st.session_state.precisions, axis=0)
                            ax.plot(mean_recall, mean_precision, color='blue', label='Mean Precision-Recall')
                            ax.set_xlabel('Recall')
                            ax.set_ylabel('Precision')
                            ax.set_title('Precision-Recall Curve for K-Fold Cross-Validation')
                            ax.legend(loc='lower left')
                            st.pyplot(fig)

                            buf = plot_to_bytes(fig)
                            st.download_button(
                                label="⬇️ Download Precision-Recall Curves",
                                data=buf,
                                file_name="precision_recall_curve_kfold.png",
                                mime="image/png"
                            )

                    elif model_type == 'Regression':
                        st.markdown("## 📊 K-Fold Cross-Validation Results for Regression")

                        # Calculate means and standard deviations
                        mean_metrics = {key: np.mean(val) for key, val in st.session_state.kf_scores.items()}
                        std_metrics = {key: np.std(val) for key, val in st.session_state.kf_scores.items()}

                        # Organizing metrics into columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Average R2", value=f"{mean_metrics['r2']:.4f}", delta=f"{std_metrics['r2']:.4f}")
                        with col2:
                            st.metric(label="Average MSE", value=f"{mean_metrics['mse']:.4f}", delta=f"{std_metrics['mse']:.4f}")
                        with col3:
                            st.metric(label="Average MAE", value=f"{mean_metrics['mae']:.4f}", delta=f"{std_metrics['mae']:.4f}")
                        
                        # Plot: Predicted vs Actual with explanation
                        st.markdown("### 📈 Best Overall Model Performance: Predicted vs. Actual Plot")
                        st.info("This plot visually inspects how well the model’s predictions align with actual values. If the points fall along the diagonal line, the model is performing well.")
                        
                        # Generating the plot
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), alpha=0.7)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title('Predicted vs Actual')
                        st.pyplot(fig)

                        # Download option for the plot
                        st.markdown("### ⬇️ Download the Plot")
                        buf = plot_to_bytes(fig)
                        st.download_button(
                            label="Download Predicted vs. Actual Plot",
                            data=buf,
                            file_name="predicted_vs_actual_kfold.png",
                            mime="image/png"
                        )

if __name__ == "__main__":
    main()
