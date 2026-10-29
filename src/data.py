import re
import numpy as np
import pandas as pd
from pathlib import Path


def read_file_with_multiple_encodings(filename: Path) -> (str, str):
    """Goes through all files in a folder and checks
    what encoding is used between utf-8 and latin-1"""
    path = Path(filename)

    for encoding in ['utf-8', 'latin-1']:
        try:
            text = path.read_text(encoding=encoding)
            return text, encoding
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not decode file")


def parse_log_files(file_content: str, pattern: str) -> list[dict]:
    """Goes through a set of transaction records and creates a dictionary for
    each transaction record and puts that into one list of dictionaries"""
    sales = []
    for line in file_content.splitlines():
        match = re.match(pattern, line)

        if match:
            timestamp = match.group(1)
            register_id = match.group(2)
            item = match.group(3)
            price = float(match.group(4))
            currency = match.group(5)
            status = match.group(6)

            sale = {
            "timestamp": timestamp,
            "register_id": register_id,
            "item": item,
            "price": price,
            "currency": currency,
            "status": status
            }
            sales.append(sale)

    return sales


def find_log_files(input_path: str | Path, path_list: list[None | Path]) -> list[Path]:
    """Goes through a given directory and creates a list of all files with
    the .log suffix in that directory and its subdirectories"""
    dir_path = Path(input_path)
    for file in dir_path.iterdir():
        if file.is_dir():
            find_log_files(file, path_list)

        else:
            if file.suffix == '.log':
                path_list.append(file)

    return path_list


def traverse_json(data: dict | list, list_for_output: list[None | str]) -> set[str]:
    """Goes through a combination of nested dictionaries and lists and
    returns a set of register id strings where 'active': True"""
    if isinstance(data, dict):
        for key in data.keys():
            if key == 'active' and data[key] is True:
                list_for_output.append(data['id'])
            traverse_json(data[key], list_for_output)

    if isinstance(data, list):
        for index in range(len(data)):
            traverse_json(data[index], list_for_output)

    return set(list_for_output)


def removing_price_outliers(input_data: pd.DataFrame) -> np.ndarray:
    """Goes through the prices in a dataframe column and replaces any values
    outside of 3 standard deviations from the mean"""
    price = [input_data["price"].iloc[i] for i in range(len(input_data))]
    arr = np.array(price)
    mean = np.mean(arr)
    std = np.std(arr)

    detect_outliers = (abs(arr - mean) > 3*std)
    median = np.median(arr[~detect_outliers])
    arr[detect_outliers] = median
    return arr


def compute_refund_ratio(sales_count: np.ndarray, refund_count: np.ndarray) -> np.ndarray:
    """Computes the element wise refund ratio for a set of transactions"""
    refund_ratio = refund_count / (sales_count + 10**-6) * 100
    return refund_ratio


def convert_to_dataframe(converting: list[dict]) -> pd.DataFrame:
    """Converts a list of dictionaries to a dataframe and sets the
    timestamp column as the index"""
    converting_df = pd.DataFrame(converting)
    converting_df['timestamp'] = pd.to_datetime(converting_df['timestamp'])
    converting_df.set_index('timestamp', inplace=True)
    sorted_df = converting_df.sort_index()
    return sorted_df


def hourly_revenue(input_df: pd.DataFrame) -> pd.Series:
    """Creates a resampled version of a dataframe grouped by item, and shows
    the revenue of that item per hour"""
    grouped = input_df.groupby('item')
    resampled = grouped['price'].resample('1h').sum()
    return resampled


def create_pivot_table(input_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a pivot table from a dataframe where the index is item,
    the column is status and the values are a sum of the prices"""
    table = pd.pivot_table(input_df, values='price', index='item', columns='status', aggfunc='sum')
    return table


def refund_ratio_per_item(input_df: pd.DataFrame) -> np.ndarray:
    """Counts succesfull and refunded transactions and calculates
    the refund ratio per item by calling the compute_refund_ratio function"""
    input_array = input_df.to_numpy()
    success_list = [i[1] for i in input_array if i[4] == "SUCCESS"]
    refund_list = [i[1] for i in input_array if i[4] == "REFUND"]

    success_array = np.array(success_list)
    success_sorts = np.sort(success_array)
    success_unique_words, success_count = np.unique(success_sorts, return_counts=True)

    refund_array = np.array(refund_list)
    refund_sorts = np.sort(refund_array)
    refund_unique_words, refund_count = np.unique(refund_sorts, return_counts=True)

    return np.vstack((success_unique_words, compute_refund_ratio(success_count, refund_count)))
