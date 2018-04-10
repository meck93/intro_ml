import pandas as pd
import numpy as np

# personal csv reader module
import reader

def count_number(array, number):
    """
    Counts the occurrence of number in array.
    """
    count = 0

    for entry in array:
        if entry == number:
            count += 1

    return count

def count_numbers(array, numbers):
    """
    Counts the occurrence of each number in array.
    """
    count = []
    
    for number in numbers:
        count.append(count_number(array, number))
    
    return count

# filename prefixes
FILE_PREFIX_XGB = "../output/xgb/task2_xgb_nativ_["
FILE_PREFIX_MLP = "../output/mlp_lbgfs/task2_mlp_lbgfs_["

# filename suffix
FILE_SUFFIX = "].csv"

# last file index e.g. [5]
INDEX_LAST_FILE = 6

# read training file
files = []

# read all existing files
for i in range(1, INDEX_LAST_FILE + 1):
    files.append(reader.read_csv(FILE_PREFIX_MLP + str(i) + FILE_SUFFIX, False))

rows = []

# create a list for all values of the same row
for i in range(0, 3000):
    row = []

    for file in files:
        row.append(file.iloc[i][1])

    rows.append(row)

average_values = []

for i in range(0, 3000):
    count = count_numbers(rows[i], [0, 1, 2])
    
    # check which value has the highest count
    if count[0] > count[1]:
        if count[0] > count[2]:
            average_values.append(0)
        else:
            average_values.append(2)
    elif count[0] > count[2]:
        average_values.append(1)
    else:
        if count[1] > count[2]:
            average_values.append(1)
        else:
            average_values.append(2)
    
    # print each row including the average value
    print("Row-Number:", i+2000,"0s:", count[0], "\t1s:", count[1], "\t2s:", count[2], "\t-overfit-value:", average_values[i])

# preparing to write the coefficients to file
out = {"Id" : files[0]['Id'], "y": average_values}

# output data frame
out = pd.DataFrame(data=out, dtype=np.int16)

# printing test output
print()
print("Result Written To File:")
print(out.head(5))

# write to csv-file
# out.to_csv("./output/xgb/average/task2_xgb_native_av[1-1].csv", sep=',', index=False, header=True)
# out.to_csv("./output/mlp_lbgfs/average/task2_mlp_lbgfs_av[1-1].csv", sep=',', index=False, header=True)
