import pandas as pd
import glob
import os

HEADERS = ['Iteration','Library','ModelName','Temperature','Strategy','Commit','File','Actual','Predicted', 'RootCause']

def concatFiles():
    # if os.path.exists('output/all_results.csv'):
    #     return False
    csv_files = glob.glob("output/*.csv")
    df = pd.concat((pd.read_csv(f, sep=',', header=None) for f in csv_files), ignore_index=True)
    df.columns = HEADERS
    return df

def normalize(df):
    df["Actual"] = df["Actual"].replace({"YES": 1, "NO": 0})
    df["Actual"] = df["Actual"].replace({True: 1})
    df["Predicted"] = df["Predicted"].replace({"YES": 1, "NO": 0})
    df["Predicted"] = df["Predicted"].replace({"BUGGY": 1})
    return df
    
def filter_results(normalized_df):
    filtered_df = normalized_df[normalized_df['Predicted'] != 'Parse Error']
    filtered_df.to_csv("output/all_results_filtered.csv", index=False, sep=',')

if __name__ == '__main__':
    df = concatFiles()
    normalized_df = normalize(df)
    filter_results(normalized_df)