import polars as pl

def pivot_table_on_column(table: pl.DataFrame, column: str, col_for_distribution: str) -> pl.DataFrame:
    """
    Pivot a Polars DataFrame to a wide format based on a categorical column.

    This function takes a DataFrame and pivots it by extracting up to 10 unique values
    from the specified `column`, then gathers the corresponding values from `col_for_distribution`
    into separate columns. The resulting DataFrame has one column per category, each containing
    the values from `col_for_distribution` where the category matched.

    If the number of values per category differs, shorter lists are padded with `None`
    to align all columns to the same length.

    Parameters
    ----------
    table : pl.DataFrame
        The input Polars DataFrame.
    column : str
        The name of the categorical column to pivot on.
    col_for_distribution : str
        The name of the column whose values will be distributed into the new columns.

    Returns
    -------
    wide_by_col : pl.DataFrame
        A wide-format DataFrame with up to 10 columns, each representing a category from `column`,
        containing values from `col_for_distribution`, padded with `None` where needed.
    col_values : list
        The list of unique category values used as columns (in the order they appear).

    Example
    -------
    >>> pivot_table_on_column(df, "Genre", "Listening_Time_minutes")
    (shape: (max_len, <=10), ['Technology', 'Health', ..., 'Comedy'])
    """

    col_values = table[column].unique().limit(15).to_list()

    data_by_col = {
        col: table.filter(pl.col(column) == col)[col_for_distribution].to_list()
        for col in col_values
    }

    # Determine the maximum length
    max_len = max(len(v) for v in data_by_col.values())

    # Pad all lists to max length with None
    padded_data = {
        col: values + [None] * (max_len - len(values))
        for col, values in data_by_col.items()
    }

    # Now create the DataFrame safely
    wide_by_col = pl.DataFrame(padded_data)
    return wide_by_col, col_values