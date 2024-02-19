import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.svm import SVR

# Load data from CSV (replace with your actual file path)
df = pd.read_csv("D:\\Github project\\Dataset Final")

# Step 1: Initial Visualization
# Example: Pairplot for numeric columns
sns.pairplot(df)
plt.show()

# Step 2: Data Cleaning (handle missing values, outliers, etc.)
# Example: Remove rows with missing values
df_cleaned = df.dropna()

# Step 3: Visualize Again (after cleaning)
# Example: Histogram for a specific column
plt.hist(df_cleaned['column_name'], bins=20)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Distribution of Column Name')
plt.show()

# Step 4: Generate a pandas profiling report
profile = ProfileReport(df_cleaned, title='Data Profiling Report', explorative=True)
profile.to_file('data_profiling_report.html')

# Now you have a cleaned dataset, a profiling report, and initial visualizations.
# Customize the steps based on your specific data and requirements.

# Step 5: SVR Modeling (use your features and target variable)
X = df_cleaned[['feature1', 'feature2']]  # Replace with your features
y = df_cleaned['target_variable']  # Replace with your target variable
svr_model = SVR(kernel='linear')  # You can choose different kernels
svr_model.fit(X, y)

# Use the trained SVR model for predictions or further analysis.
