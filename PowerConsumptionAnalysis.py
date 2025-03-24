import pandas as pd              
import numpy as np               
import statsmodels.api as sm    
import matplotlib.pyplot as plt  
import seaborn as sns            
import sys
from persiantools.jdatetime import JalaliDate  
from sklearn.model_selection import train_test_split
sys.stdout.reconfigure(encoding='utf-8')

# ---------------------- 
file_path = r"C:\Users\Lenovo\Desktop\analysis.xlsx" 

# Read the Excel file
df = pd.read_excel(file_path)
print("Initial Data:")
print(df.head())
print(df.info())

# ---------------------- 

def persian_to_english(text):
    if pd.isna(text):
        return text
    if isinstance(text, str):
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        english_digits = "0123456789"
        translation_table = str.maketrans(persian_digits, english_digits)
        return text.translate(translation_table)
    else:
        return text

if "from date" in df.columns:
    df["from date"] = df["from date"].apply(persian_to_english)
if "to date" in df.columns:
    df["to date"] = df["to date"].apply(persian_to_english)

df_cleaned = df.dropna().copy()
print("\nMissing values count after dropping NaN rows:")
print(df_cleaned.isnull().sum())
print("\nData types in each column:")
print(df_cleaned.dtypes)

# ---------------------- 

def convert_shamsi_to_miladi(date_str):
    try:
        parts = date_str.split('/')
        if len(parts) != 3:
            raise ValueError("Incorrect date format")
        year, month, day = map(int, parts)
        greg_date = JalaliDate(year, month, day).to_gregorian()
        return pd.to_datetime(greg_date)
    except Exception as e:
        print(f"Error converting date {date_str}: {e}")
        return pd.NaT

df_cleaned["from date"] = df_cleaned["from date"].apply(lambda x: convert_shamsi_to_miladi(x) if isinstance(x, str) else x)
df_cleaned["to date"] = df_cleaned["to date"].apply(lambda x: convert_shamsi_to_miladi(x) if isinstance(x, str) else x)

df_cleaned = df_cleaned.dropna(subset=["from date", "to date"])

# ---------------------- 

reference_date = df_cleaned["from date"].min()

# Create new columns representing the number of days since the reference date
df_cleaned["from date numeric"] = (df_cleaned["from date"] - reference_date).dt.days
df_cleaned["to date numeric"] = (df_cleaned["to date"] - reference_date).dt.days

print("\nData after converting dates:")
print(df_cleaned.head())


feature_cols = ['number of days', 'average consumptoin', 'average daily consumptoin', 'from date numeric']
target_col = 'total consumptoin'

missing_cols = [col for col in feature_cols + [target_col] if col not in df_cleaned.columns]
if missing_cols:
    raise KeyError(f"The following columns are missing in the dataset: {missing_cols}")

X = df_cleaned[feature_cols]
y = df_cleaned[target_col]

print("\nShapes of features and target:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------------- 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShapes of the split data:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.legend()
plt.title("Comparison of Predicted vs Actual Consumption")
plt.show()



# ---------------------- 

future_days = 90  

future_dates = pd.date_range(start=df_cleaned["to date"].max(), periods=future_days, freq='D')

# تبدیل تاریخ‌های آینده به عدد
future_numeric_dates = (future_dates - reference_date).days

avg_num_days = df_cleaned["number of days"].mean()
avg_consumption = df_cleaned["average consumptoin"].mean()
avg_daily_consumption = df_cleaned["average daily consumptoin"].mean()

future_data = pd.DataFrame({
    'number of days': [avg_num_days] * future_days,
    'average consumptoin': [avg_consumption] * future_days,
    'average daily consumptoin': [avg_daily_consumption] * future_days,
    'from date numeric': future_numeric_dates
})

future_predictions = model.predict(future_data)


future_data["Predicted Consumption"] = future_predictions
future_data["Date"] = future_dates

# ---------------------- 
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned["to date"], df_cleaned["total consumptoin"], label="Actual Consumption", marker='o')
plt.plot(future_data["Date"], future_data["Predicted Consumption"], label="Predicted Future Consumption", linestyle="dashed", marker='s')

plt.xlabel("Date")
plt.ylabel("Total Consumption")
plt.title("Predicted Electricity Consumption for the Next 3 Months")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
