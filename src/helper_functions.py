import re


def calc_age_in_months(val):
    # If NaN, return NaN
    if val != val:
        return val
    # Extract numeric values using regex
    pattern = re.compile(r'\d+')
    digits = pattern.findall(val)
    # Convert to integer and perform calculation
    result = int(digits[0]) * 12 + int(digits[1])
    return result


def clean_outliers(data, column_name: str):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)

    IQR = Q3 - Q1

    data = data.drop(data.loc[data[column_name] > (Q3 + 1.5 * IQR)].index)
    data = data.drop(data.loc[data[column_name] < (Q1 - 1.5 * IQR)].index)

    return data
