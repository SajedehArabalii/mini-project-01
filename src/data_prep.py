""" 
Preprocessing of data 
    - loading 
    - checking 
    - cleaning 
    - splitting 
    - scaling 
""" 
import pandas as pd 
from pathlib import Path 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
 
def prepare_data(): 
    ROOT = Path(__file__).resolve().parent.parent 
    DATA_PATH = ROOT / "data" / "creditcard.csv" 
    df = pd.read_csv(DATA_PATH) 
 
    # # Check data type 
    # print(f"data types: {df.dtypes}") 
    # # Number of samples and features 
    # print(f"samples x features : {df.shape}") 
    # # Class distribution 
    # print(f"class distribution : \n{df['Class'].value_counts()}") 
    # # Number of missing values 
    # print(f"missing values : \n{df.isna().sum()}") 
    # # Check for duplicates 
    # print(f"duplicate : {df.duplicated().sum()}") 
    # # Descriptive Statistics 
    # print(f"Discriptive Statistics : {df.describe()}") 
 
    # Remove duplicates 
    df = df.drop_duplicates() 
 
    # Split your features and target 
    X = df.drop("Class", axis=1) 
    y = df["Class"] 
 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, stratify=y, random_state=2 
    ) 
 
    # feature scaling 
    scaler = StandardScaler() 
    scaler.fit(X_train) 
    X_train_scaled = scaler.transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
 
    return ( 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
 
    ) 
 
