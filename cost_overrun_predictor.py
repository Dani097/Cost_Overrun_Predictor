# ============================================================
# PROJECT: Construction Cost Overrun Predictor
# AUTHOR:  [Your Name]
# PURPOSE: Predict whether a construction project will go
#          over budget using machine learning (regression).
#          This bridges civil engineering knowledge with
#          data science and financial technology.
# ============================================================

# --- STEP 1: IMPORT LIBRARIES ---
# Think of libraries as toolboxes. We import them so we can
# use their pre-built tools instead of writing everything ourselves.

import pandas as pd               # pandas = tool for handling tables of data (like Excel in Python)
import numpy as np                # numpy  = tool for math and number operations
import matplotlib.pyplot as plt   # matplotlib = tool for drawing charts and graphs
from sklearn.linear_model import LinearRegression   # our machine learning model
from sklearn.model_selection import train_test_split # splits data into training and testing sets
from sklearn.metrics import mean_absolute_error, r2_score  # tools to measure how accurate our model is


# --- STEP 2: CREATE SAMPLE DATA ---
# In a real project, you would load this from a CSV file.
# Here we create fictional but realistic construction project data
# so the code works on its own without any external files.
#
# Each row = one past construction project
# Columns = features (inputs) that affect cost

# np.random.seed(42) makes sure we get the same "random" numbers
# every time we run the code. This makes results reproducible.
np.random.seed(42)

# Number of past projects in our dataset
num_projects = 100

# --- Generate realistic input features ---

# Project duration in months (between 6 and 60 months)
duration_months = np.random.randint(6, 60, num_projects)

# Original estimated budget in £ thousands (between £500k and £10 million)
estimated_budget_k = np.random.randint(500, 10000, num_projects)

# Number of contractors involved (1 to 10)
num_contractors = np.random.randint(1, 10, num_projects)

# Project type encoded as a number:
# 0 = Residential, 1 = Commercial, 2 = Infrastructure, 3 = Industrial
project_type = np.random.randint(0, 4, num_projects)

# Site complexity score (1 = simple, 10 = very complex)
# Civil engineers know that ground conditions, access, etc. affect cost
site_complexity = np.random.randint(1, 10, num_projects)

# --- Generate the TARGET variable (what we want to predict) ---
# Cost overrun % = how much over budget the project went
# We simulate this using a formula that reflects real-world logic:
#   - Longer projects tend to overrun more
#   - Higher complexity = more overrun
#   - More contractors = more coordination issues = more overrun
# We also add some random "noise" to make the data realistic

cost_overrun_percent = (
    0.3 * duration_months        # longer = more overrun
    + 0.002 * estimated_budget_k # bigger budget = slightly more overrun
    + 1.5 * num_contractors      # more contractors = more risk
    + 2.0 * site_complexity      # more complex = more overrun
    + np.random.normal(0, 5, num_projects)  # random noise (unexpected events)
)

# --- Assemble everything into a DataFrame (a table) ---
# A DataFrame is like an Excel spreadsheet inside Python
data = pd.DataFrame({
    'duration_months':    duration_months,
    'estimated_budget_k': estimated_budget_k,
    'num_contractors':    num_contractors,
    'project_type':       project_type,
    'site_complexity':    site_complexity,
    'cost_overrun_%':     cost_overrun_percent
})

# Print the first 5 rows so we can see what our data looks like
print("=" * 60)
print("SAMPLE DATA (first 5 projects):")
print("=" * 60)
print(data.head())
print()


# --- STEP 3: EXPLORE THE DATA ---
# Before building a model, we should understand what the data looks like.
# This is called Exploratory Data Analysis (EDA).

print("=" * 60)
print("BASIC STATISTICS OF OUR DATASET:")
print("=" * 60)
print(data.describe())  # shows min, max, average, etc. for each column
print()


# --- STEP 4: PREPARE DATA FOR MACHINE LEARNING ---
# We need to split our data into:
#   X = input features (the things we know about a project)
#   y = output / target (the cost overrun % we want to predict)

# X = all columns EXCEPT the cost overrun column
X = data.drop(columns=['cost_overrun_%'])

# y = just the cost overrun column
y = data['cost_overrun_%']

# Split: 80% of data for TRAINING the model, 20% for TESTING it
# This is like studying with 80% of past exam papers, then testing
# yourself with the remaining 20% you've never seen before.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} projects")
print(f"Testing set size:  {len(X_test)} projects")
print()


# --- STEP 5: TRAIN THE MACHINE LEARNING MODEL ---
# We use Linear Regression — the simplest ML model.
# It finds a mathematical line/formula that best fits the data.
# Think of it like drawing the best-fit line on a scatter plot.

model = LinearRegression()  # create the model object
model.fit(X_train, y_train) # train it using our training data

print("=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)


# --- STEP 6: EVALUATE THE MODEL ---
# Now we test how accurate the model is on the 20% it has never seen.

y_predicted = model.predict(X_test)  # model makes predictions

# MAE = Mean Absolute Error: on average, how many % points off is our prediction?
mae = mean_absolute_error(y_test, y_predicted)

# R² score: how much of the variation does our model explain?
# 1.0 = perfect, 0.0 = no better than guessing the average
r2 = r2_score(y_test, y_predicted)

print(f"Mean Absolute Error (MAE): {mae:.2f}%")
print(f"R² Score:                  {r2:.2f}  (1.0 = perfect)")
print()


# --- STEP 7: PREDICT A NEW PROJECT ---
# This is the practical part — a real user enters project details
# and the model predicts the expected cost overrun.

print("=" * 60)
print("PREDICT COST OVERRUN FOR A NEW PROJECT")
print("=" * 60)

# Example: A new bridge project
# In a real app, these values would come from user input
new_project = pd.DataFrame({
    'duration_months':    [36],    # 3-year project
    'estimated_budget_k': [5000],  # £5 million budget
    'num_contractors':    [6],     # 6 contractors involved
    'project_type':       [2],     # 2 = Infrastructure
    'site_complexity':    [8],     # fairly complex site
})

# Use the trained model to predict
predicted_overrun = model.predict(new_project)[0]

# Calculate the financial impact
budget = new_project['estimated_budget_k'][0] * 1000  # convert to £
extra_cost = budget * (predicted_overrun / 100)

print(f"Project Duration:       {new_project['duration_months'][0]} months")
print(f"Estimated Budget:       £{budget:,.0f}")
print(f"Predicted Cost Overrun: {predicted_overrun:.1f}%")
print(f"Expected Extra Cost:    £{extra_cost:,.0f}")
print()

# Flag financial risk level based on overrun %
if predicted_overrun < 10:
    risk_level = "LOW RISK ✅"
elif predicted_overrun < 25:
    risk_level = "MEDIUM RISK ⚠️"
else:
    risk_level = "HIGH RISK 🚨"

print(f"Financial Risk Level:   {risk_level}")
print()


# --- STEP 8: VISUALISE THE RESULTS ---
# Good data scientists always visualise their results.
# We'll create two charts: one showing model accuracy,
# one showing which factors matter most.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Construction Cost Overrun Predictor', fontsize=16, fontweight='bold')

# --- Chart 1: Actual vs Predicted Values ---
# If our model is perfect, all dots would lie on the diagonal line.
# The closer the dots are to the line, the better the model.
ax1.scatter(y_test, y_predicted, color='steelblue', alpha=0.6, edgecolors='white', s=60)
ax1.plot(                                    # draw the "perfect prediction" line
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', linewidth=2, label='Perfect Prediction'
)
ax1.set_xlabel('Actual Cost Overrun (%)', fontsize=12)
ax1.set_ylabel('Predicted Cost Overrun (%)', fontsize=12)
ax1.set_title(f'Actual vs Predicted\n(R² = {r2:.2f}, MAE = {mae:.1f}%)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Chart 2: Feature Importance ---
# Which factors have the most influence on cost overrun?
# In linear regression, larger coefficients = more influence.
feature_names = X.columns.tolist()
coefficients = model.coef_  # the model's learned weights for each feature

# Sort by absolute size so the most important features appear at the top
sorted_indices = np.argsort(np.abs(coefficients))[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_coefs = [coefficients[i] for i in sorted_indices]

# Use different colours: orange = positive effect, green = negative effect
colors = ['coral' if c > 0 else 'mediumseagreen' for c in sorted_coefs]

ax2.barh(sorted_features, sorted_coefs, color=colors, edgecolor='white')
ax2.set_xlabel('Coefficient (Impact on Overrun %)', fontsize=12)
ax2.set_title('What Drives Cost Overrun?\n(Feature Importance)', fontsize=12)
ax2.axvline(x=0, color='black', linewidth=0.8)  # vertical line at zero
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cost_overrun_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Charts saved! Analysis complete.")
print("=" * 60)
