import pandas as pd

def read_csv(file_path):
    # Read the data into a data frame
    data = pd.read_csv(file_path)

    # Check the number of data points in the data set
    print("# of data points (rows):", len(data))

    # Check the number of features in the data set
    print("# of features (columns):", len(data.columns))

    # Check the data types
    # print(data.dtypes.unique())

    # Check any number of columns with NaN
    # print("NaN in Rows:", data.isnull().any().sum(), ' / ', len(data.columns))

    # Check any number of data points with NaN
    # print("NaN in Columns:", data.isnull().any(axis=1).sum(), ' / ', len(data))

    # Return the data frame
    return data