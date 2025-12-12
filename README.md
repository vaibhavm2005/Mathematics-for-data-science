# Mathematics-for-data-science
## Project Overview
This repository contains two core supervised machine learning projects developed in Python using Scikit-learn, demonstrating end-to-end expertise in predictive modeling:
1) **Linear Regression**: Predicting a continuous numerical value (Annual Customer Spend).
2) **Logistic Regression**: Predicting a binary categorical outcome (User Ad Click).

The objective is to not only build accurate models but also to interpret the results to provide clear, actionable business recommendations.

## Project Structure
* `06-Linear Regression Project.ipynb`: Full pipeline for customer spending prediction.
* `07-Logistic Regression Project.ipynb`: Full pipeline for binary ad click classification.
* `Ecommerce Customers`, `advertising.csv`: Dataset files.

## Core Models and Insights
### 1) Linear Regression: Annual Customer Spend Prediction
This model aimed to advise an e-commerce company on whether to focus investment on their mobile app or their website to drive revenue.
* **Problem Type**: Continuous Regression
* **Model**: `LinearRegression`
* **Evaluation**: Mean Squared Error (MSE), Residual Analysis.
* **Key Finding**: Coefficient analysis demonstrated that 'Length of Membership' had the strongest positive correlation with 'Yearly Amount Spent,' indicating that customer retention is the most significant revenue driver.
* **Recommendation**: Advised stakeholders to prioritize strategies for increasing customer loyalty and retention over immediate feature development.
### 2) Logistic Regression: Ad Click Prediction
This model was developed to predict whether a user would click on a digital advertisement, enabling more efficient targeted marketing.
* **Problem Type**: Binary Classification (0/1)
* **Model**: `LogisticRegression`
* **Evaluation**: Classification Report (Precision, Recall, F1-Score), Confusion Matrix.
* **Key Finding**: Achieved 93% overall accuracy on the test set. Analysis showed high Precision (low False Positives), ensuring the model is reliable for targeted campaigns.
* **Recommendation**: EDA revealed an inverse correlation: users spending less time on the site were more likely to click the ad, suggesting successful targeting of the non-loyal/casual segment.

## Technical Skills Demonstrated
* **Machine Learning:** Linear Regression, Logistic Regression, Supervised Learning, Model Training & Validation.
* **Evaluation Metrics:** Interpreting coefficients, diagnosing model fit using Residuals, Confusion Matrix, Precision, Recall, F1-Score, and Support.
* **Data Analysis (EDA):** Feature selection, visualization of correlations using Seaborn (pairplot), data scaling (StandardScaler).
* **Python Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
* **Engineering Practices:** Code modularity, clear variable naming, setting random_state for reproducibility.

## Setup and Dependencies
### Option 1: Local Setup
To run these notebooks locally, you will need a Python environment with the following libraries installed.
```python
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
#### Running the Project
1) Clone this repository:
```python
git clone [Your Actual GitHub Link Here]
cd [Your Repository Name]
```
2) Launch Jupyter Notebook:
```python
jupyter notebook
```
3) Open and run the two `.ipynb` files sequentially.

### Option 2: Running in Google Colab (Recommended)
You can run these notebooks directly in Google Colab, which handles the environment setup and dependencies automatically.
#### Open Notebooks in colab: 
* Linear Regression:
* Logistic Regression:
#### Steps in Colab:
1) Click the link above for the desired notebook.
2) **Clone the Repository and Install Dependencies:** In the first code cell of the notebook, run the following commands:
```python
# Clone the repository
!git clone [Your Actual GitHub Link Here]

# Change into the project directory
%cd [Your Repository Name]

# Install dependencies (only required the first time)
!pip install pandas numpy scikit-learn matplotlib seaborn
```
3) Upload the dataset files.
4) Run the subsequent cells in the notebook.
















