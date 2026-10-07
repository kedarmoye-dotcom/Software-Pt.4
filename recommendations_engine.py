"""
recommendations_engine.py -- Part 4: Recommendations

Entry point for the rest of the platform:
    render_recommendations(df, quality_results=None, analytics_results=None)

The engine reads the DataFrame (never modifies it), applies transparent
threshold rules, and produces a prioritized table of recommendations.

Optional structured inputs from earlier parts:
    quality_results   = {"invalid_values": [{"column": str, "issue": str,
                         "count": int}, ...]}   (list of dicts or DataFrame)
    analytics_results = {"correlation_matrix": pandas.DataFrame}
Both are optional; missing keys simply mean the engine runs its own checks.
"""

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
# warnings: silences pandas' date-format notices while testing whether text
#           columns contain dates.
import warnings

# numpy: detects infinite values and reads the correlation matrix as an array.
import numpy as np

# pandas: all statistics (missing counts, duplicates, quartiles, skewness,
#         correlations) and the final recommendation table.
import pandas as pd

# streamlit: renders every heading, message, control, table and download.
import streamlit as st

# Reuse the Analytics (Part 4) helpers so column detection is consistent
# across the platform and logic is not duplicated.
from analytics import (detect_datetime_columns, detect_numeric_columns,
                       make_unique_columns)


# ---------------------------------------------------------
# PRIORITY LABELS AND DISPLAY ORDER
# ---------------------------------------------------------
HIGH, MEDIUM, LOW, INFO = "High", "Medium", "Low", "Informational"
PRIORITY_ORDER = [HIGH, MEDIUM, LOW, INFO]

# Headings shown on the "By Priority" tab for each priority level.
PRIORITY_HEADINGS = {HIGH: "High Priority Recommendations",
                     MEDIUM: "Medium Priority Recommendations",
                     LOW: "Low Priority Recommendations",
                     INFO: "Informational Notes"}

# Category names, grouped for the Data Quality and Analytics tabs.
DATA_QUALITY_CATEGORIES = ["Data Quality", "Missing Data", "Duplicates",
                           "Outliers", "Data Types", "Categorical Data"]
ANALYTICS_CATEGORIES = ["Numerical Analysis", "Correlation",
                        "Dataset Structure", "General Recommendations"]
CATEGORY_ORDER = DATA_QUALITY_CATEGORIES + ANALYTICS_CATEGORIES

# Column order of the recommendation table (also the CSV export layout).
RECOMMENDATION_COLUMNS = ["Priority", "Category", "Issue", "Finding",
                          "Recommendation", "Suggested Action",
                          "Affected Columns", "Rule"]

# ---------------------------------------------------------
# TRANSPARENT RULE THRESHOLDS (edit these to change the rules)
# ---------------------------------------------------------
# Missing values: share of a column's rows that are empty.
HIGH_MISSING_THRESHOLD = 0.50      # >= 50% missing -> High
MEDIUM_MISSING_THRESHOLD = 0.20    # >= 20% missing -> Medium (otherwise Low)

# Invalid/placeholder values: share of a column's rows affected.
HIGH_INVALID_THRESHOLD = 0.20      # >= 20% affected -> High
MEDIUM_INVALID_THRESHOLD = 0.05    # >= 5% affected  -> Medium (otherwise Low)

# Duplicate rows: share of all rows that repeat an earlier row.
HIGH_DUPLICATE_THRESHOLD = 0.10    # >= 10% duplicates -> High
MEDIUM_DUPLICATE_THRESHOLD = 0.01  # >= 1% duplicates  -> Medium (otherwise Low)

# Outliers: IQR fence multiplier and the share of values outside the fences.
IQR_MULTIPLIER = 1.5               # fences = Q1 - 1.5*IQR and Q3 + 1.5*IQR
MEDIUM_OUTLIER_THRESHOLD = 0.05    # >= 5% outside fences -> Medium (else Low)

# Minimum number of non-missing values before statistics are trusted.
MIN_ROWS_FOR_STATS = 8

# Data types: share of sampled text values that must convert cleanly.
TEXT_CONVERSION_SHARE = 0.90
TYPE_SAMPLE_SIZE = 2000            # max values sampled per text column
BOOLEAN_TOKENS = {"yes", "no", "y", "n", "true", "false", "t", "f"}
PLACEHOLDER_TOKENS = {"", "n/a", "na", "null", "none", "nan", "unknown", "?",
                      "-", "--", "missing", "#n/a"}

# Categorical columns.
HIGH_CARDINALITY_COUNT = 50        # more distinct values than this is "high"
ID_LIKE_RATIO = 0.90               # distinct/non-null >= 90% looks like an ID
ID_LIKE_MIN_UNIQUE = 20            # ...but only with more than 20 distinct values
RARE_CATEGORY_THRESHOLD = 0.01     # category below 1% of rows is "rare"
RARE_MIN_ROWS = 100                # rare-category check needs >= 100 rows

# Numerical columns.
NEAR_CONSTANT_SHARE = 0.95         # one value in >= 95% of rows
SKEW_THRESHOLD = 1.0               # |skewness| >= 1 is "highly skewed"
EXTREME_RANGE_RATIO = 50           # (max - min) / IQR >= 50 is a wide range

# Correlations (absolute value of the coefficient).
STRONG_CORRELATION = 0.70          # >= 0.70 -> Informational
REDUNDANT_CORRELATION = 0.95       # >= 0.95 -> Low (possible redundancy)

# Dataset structure.
VERY_SMALL_ROWS = 10               # < 10 rows -> High
SMALL_ROWS = 30                    # < 30 rows -> Medium
WIDE_DATASET_COLUMNS = 100         # > 100 columns -> Low

# How many items a single finding lists before summarizing the rest.
MAX_LISTED_ITEMS = 8

# Prefix for widget keys so they never clash with other platform parts.
KEY = "recommendations"


# ---------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------
def make_recommendation(priority, category, issue, finding, recommendation,
                        action, rule, columns=()):
    """Build one recommendation record (a dict matching RECOMMENDATION_COLUMNS)."""
    return {"Priority": priority, "Category": category, "Issue": issue,
            "Finding": finding, "Recommendation": recommendation,
            "Suggested Action": action,
            "Affected Columns": ", ".join(str(c) for c in columns),
            "Rule": rule}


def describe_items(items, limit=MAX_LISTED_ITEMS):
    """Join items into a readable sentence fragment, summarizing the overflow."""
    text = "; ".join(items[:limit])
    if len(items) > limit:
        text += f"; and {len(items) - limit} more"
    return text


def is_text_column(series):
    """True for object/string columns (not categorical, boolean or numeric)."""
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def column_kind(series):
    """Classify a column as numeric, datetime, categorical or other."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if is_text_column(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    return "other"


def finite_values(series):
    """Non-missing, non-infinite values of a numeric Series."""
    values = series.dropna()
    return values[~np.isinf(values.astype(float))]


def group_entries_by_tier(entries, high_threshold, medium_threshold):
    """Split (column, text, share, extra) entries into High/Medium/Low lists."""
    grouped = {HIGH: [], MEDIUM: [], LOW: []}
    for entry in entries:
        share = entry[2]
        if share >= high_threshold:
            grouped[HIGH].append(entry)
        elif share >= medium_threshold:
            grouped[MEDIUM].append(entry)
        else:
            grouped[LOW].append(entry)
    return grouped


def sample_text(series):
    """Deterministic sample of stripped string values from a text column."""
    values = series.dropna()
    values = values[values.map(lambda v: isinstance(v, str))]
    if values.empty:
        return values
    size = min(len(values), TYPE_SAMPLE_SIZE)
    return values.sample(size, random_state=0).str.strip()


def numeric_text_share(series):
    """Share of sampled text values that convert to numbers (0 if not text)."""
    if not is_text_column(series):
        return 0.0
    values = sample_text(series)
    if values.empty:
        return 0.0
    # Remove thousands separators, currency and percent symbols before testing.
    cleaned = values.str.replace(r"[,$%\s]", "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def date_text_share(series):
    """Share of sampled text values that parse as dates (0 if not date-like)."""
    if not is_text_column(series) or numeric_text_share(series) >= 0.5:
        return 0.0
    values = sample_text(series)
    # Real dates contain digits; this avoids parsing ordinary words.
    values = values[values.str.contains(r"\d", regex=True)]
    if values.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values, errors="coerce")
    return float(parsed.notna().mean())


def column_names(entries):
    """Column names from (column, text, share, extra) entries."""
    return [entry[0] for entry in entries]


# ---------------------------------------------------------
# ANALYZERS: each returns a list of recommendation records
# ---------------------------------------------------------
def analyze_missing_values(df):
    """Missing-value recommendations, tiered by the share missing per column."""
    total_rows = len(df)
    entries = []
    for column, count in df.isna().sum().items():
        if count > 0:
            share = count / total_rows
            kind = column_kind(df[column])
            entries.append((column, f"{column}: {count:,} missing "
                            f"({share:.1%}, {kind})", share, kind))
    grouped = group_entries_by_tier(entries, HIGH_MISSING_THRESHOLD,
                                    MEDIUM_MISSING_THRESHOLD)

    # Treatment ideas depend on the data type of the affected columns.
    hints = {
        "numeric": "For numeric columns, consider median or mean imputation.",
        "categorical": "For text/category columns, consider a placeholder "
                       "such as 'Unknown' or the most frequent value.",
        "datetime": "For date columns, verify the source first; imputed dates "
                    "can distort trends.",
    }
    wording = {
        HIGH: ("High share of missing values",
               "Potential issue: these columns are missing a large share of "
               "their values. Investigate why before deciding whether the "
               "column is usable.",
               f"High when a column is missing >= {HIGH_MISSING_THRESHOLD:.0%} "
               "of its values."),
        MEDIUM: ("Moderate share of missing values",
                 "These columns have a noticeable share of missing values "
                 "that may bias results. Consider investigating the source.",
                 f"Medium when the missing share is >= "
                 f"{MEDIUM_MISSING_THRESHOLD:.0%} and below "
                 f"{HIGH_MISSING_THRESHOLD:.0%}."),
        LOW: ("Small share of missing values",
              "A small share of values is missing. Consider whether it is "
              "random or systematic.",
              f"Low when the missing share is below "
              f"{MEDIUM_MISSING_THRESHOLD:.0%}."),
    }
    recs = []
    for priority, tier_entries in grouped.items():
        if not tier_entries:
            continue
        issue, recommendation, rule = wording[priority]
        kinds = sorted({e[3] for e in tier_entries})
        action = ("Investigate the source of the gaps, then choose to impute, "
                  "remove affected records, or keep them. "
                  + " ".join(hints[k] for k in kinds if k in hints)
                  + " Keep missing values when the absence itself is "
                  "meaningful.")
        recs.append(make_recommendation(
            priority, "Missing Data", issue,
            describe_items([e[1] for e in tier_entries]), recommendation,
            action, rule, column_names(tier_entries)))
    return recs


def analyze_invalid_values(df, quality_results=None):
    """Placeholder text, infinite numbers and invalid values from Part 3."""
    total_rows = len(df)
    entries = []

    for column in df.columns:
        series = df[column]
        # Text values that often stand in for "no data" (e.g. 'N/A', '?').
        if is_text_column(series):
            text = series.dropna().astype(str).str.strip().str.lower()
            count = int(text.isin(PLACEHOLDER_TOKENS).sum())
            if count:
                entries.append((column, f"{column}: {count:,} possible "
                                "placeholder text values",
                                count / total_rows, None))
        # Infinite numeric values usually come from failed calculations.
        elif pd.api.types.is_float_dtype(series):
            count = int(np.isinf(series.dropna()).sum())
            if count:
                entries.append((column, f"{column}: {count:,} infinite values",
                                count / total_rows, None))

    # Invalid values reported by the Data Validation part, when supplied.
    reported = (quality_results or {}).get("invalid_values")
    if isinstance(reported, pd.DataFrame):
        reported = reported.to_dict("records")
    for item in reported or []:
        if isinstance(item, dict) and {"column", "issue", "count"} <= set(item):
            try:
                count = int(item["count"])
            except (TypeError, ValueError):
                continue
            if count > 0:
                entries.append((item["column"], f"{item['column']}: {count:,} "
                                f"values flagged ({item['issue']})",
                                count / total_rows, None))

    grouped = group_entries_by_tier(entries, HIGH_INVALID_THRESHOLD,
                                    MEDIUM_INVALID_THRESHOLD)
    rules = {HIGH: f"High when >= {HIGH_INVALID_THRESHOLD:.0%} of a column's "
                   "rows are affected.",
             MEDIUM: f"Medium when >= {MEDIUM_INVALID_THRESHOLD:.0%} (and "
                     f"below {HIGH_INVALID_THRESHOLD:.0%}) are affected.",
             LOW: f"Low when fewer than {MEDIUM_INVALID_THRESHOLD:.0%} are "
                  "affected."}
    recs = []
    for priority, tier_entries in grouped.items():
        if tier_entries:
            recs.append(make_recommendation(
                priority, "Data Quality",
                "Possible invalid or placeholder values",
                describe_items([e[1] for e in tier_entries]),
                "Consider reviewing these values. Some may be legitimate "
                "(for example, 'Unknown' can be a real category), while "
                "others may represent missing or invalid data.",
                "Review the flagged values with someone who knows the data, "
                "then decide whether to convert them to missing values, "
                "correct them, or keep them.",
                rules[priority], column_names(tier_entries)))
    return recs


def analyze_duplicates(df):
    """Exact duplicate-row recommendation, tiered by the share of duplicates."""
    try:
        duplicate_mask = df.duplicated(keep="first")
    except TypeError:
        # Columns holding lists/dicts cannot be compared for duplicates.
        return [make_recommendation(
            INFO, "Duplicates", "Duplicate check could not be completed",
            "Some columns contain unhashable values (such as lists).",
            "Duplicate detection was skipped for this dataset.",
            "Convert list-like columns to text and re-run the check.",
            "Informational whenever a check cannot be performed.")]
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        return []

    share = duplicate_count / len(df)
    groups = len(df[df.duplicated(keep=False)].drop_duplicates())
    if share >= HIGH_DUPLICATE_THRESHOLD:
        priority = HIGH
    elif share >= MEDIUM_DUPLICATE_THRESHOLD:
        priority = MEDIUM
    else:
        priority = LOW

    # If every column has low variety, repeated rows may be genuine repeated
    # observations rather than accidental copies.
    low_variety = all(df[c].nunique(dropna=False) / len(df) < 0.5
                      for c in df.columns)
    caveat = ("Every column has limited variety, so repeated rows may be "
              "legitimate repeated observations."
              if low_variety else
              "No identifier or timestamp can be assumed to separate these "
              "rows, so they may be accidental copies or legitimate "
              "repeated observations.")
    return [make_recommendation(
        priority, "Duplicates", "Duplicate records detected",
        f"{duplicate_count:,} rows ({share:.1%}) match an earlier row in "
        f"every column (exact duplicates), forming {groups:,} repeated "
        f"record(s). {caveat}",
        "Review the duplicate records and determine whether they are "
        "duplicate entries or legitimate repeated observations. Duplicates "
        "can inflate counts and bias statistics.",
        "Filter the rows that repeat another row, confirm with the data "
        "owner, and remove them only after you have confirmed they are "
        "true duplicates.",
        f"High when duplicates >= {HIGH_DUPLICATE_THRESHOLD:.0%} of rows; "
        f"Medium when >= {MEDIUM_DUPLICATE_THRESHOLD:.0%}; otherwise Low.",
        list(df.columns))]


def analyze_outliers(df, numeric_columns):
    """Potential-outlier recommendations using the IQR rule."""
    entries = []
    for column in numeric_columns:
        values = finite_values(df[column])
        if len(values) < MIN_ROWS_FOR_STATS:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        count = int(((values < lower) | (values > upper)).sum())
        if count:
            share = count / len(values)
            entries.append((column, f"{column}: {count:,} potential outliers "
                            f"({share:.1%}) outside {lower:,.2f} to "
                            f"{upper:,.2f}", share, None))

    # Outliers are never rated High: extreme values are often valid.
    grouped = group_entries_by_tier(entries, float("inf"),
                                    MEDIUM_OUTLIER_THRESHOLD)
    rules = {MEDIUM: f"Medium when >= {MEDIUM_OUTLIER_THRESHOLD:.0%} of a "
                     f"column's values fall outside the {IQR_MULTIPLIER} x IQR "
                     "fences. Never High, because outliers may be valid.",
             LOW: "Low when fewer values fall outside the IQR fences."}
    recs = []
    for priority in (MEDIUM, LOW):
        if grouped[priority]:
            recs.append(make_recommendation(
                priority, "Outliers", "Potential outliers detected",
                describe_items([e[1] for e in grouped[priority]]),
                "Investigate potential outliers before removing them, "
                "because extreme observations may represent valid business "
                "or scientific events.",
                "Inspect the rows with extreme values, compare them with the "
                "source, and decide whether to keep, correct or exclude "
                "them. Consider comparing results with and without them.",
                rules[priority], column_names(grouped[priority])))
    return recs


def analyze_data_types(df):
    """Possible data-type inconsistencies in text and mixed-type columns."""
    numbers_as_text, dates_as_text, mixed = [], [], []
    inconsistent_bool, bool_like = [], []
    for column in df.columns:
        series = df[column]
        if not is_text_column(series):
            continue
        # Columns holding more than one Python type (e.g. numbers and text).
        type_names = sorted(series.dropna().map(lambda v: type(v).__name__)
                            .unique())
        if len(type_names) > 1:
            mixed.append((column, f"{column} ({', '.join(type_names)})"))
            continue
        number_share = numeric_text_share(series)
        if number_share >= TEXT_CONVERSION_SHARE:
            numbers_as_text.append((column, f"{column} ({number_share:.0%} of "
                                    "sampled values look numeric)"))
            continue
        date_share = date_text_share(series)
        if date_share >= TEXT_CONVERSION_SHARE:
            dates_as_text.append((column, f"{column} ({date_share:.0%} of "
                                  "sampled values look like dates)"))
            continue
        values = sample_text(series)
        lowered = values.str.lower()
        if lowered.nunique() >= 2 and set(lowered.unique()) <= BOOLEAN_TOKENS:
            spellings = values.nunique()
            (inconsistent_bool if spellings > 2 else bool_like).append(
                (column, f"{column} ({spellings} spellings)"))

    share_text = f"{TEXT_CONVERSION_SHARE:.0%}"
    # (priority, issue, found items, recommendation, action, rule)
    specs = [
        (MEDIUM, "Numbers possibly stored as text", numbers_as_text,
         "Standardize the data type of these columns before further "
         "analysis; as text they are excluded from numeric statistics, "
         "histograms and correlations.",
         "Review the values, then convert the columns to numeric types "
         "(after removing symbols such as $ or commas).",
         f"Medium when >= {share_text} of sampled text values convert to "
         "numbers."),
        (MEDIUM, "Dates possibly stored as text", dates_as_text,
         "Standardize these columns as date/time types before time-series "
         "analysis so they can be sorted and aggregated correctly.",
         "Validate the date format, then convert the columns to date/time "
         "values.",
         f"Medium when >= {share_text} of sampled text values parse as "
         "dates."),
        (MEDIUM, "Mixed data types in a column", mixed,
         "Possible inconsistency: these columns hold more than one kind of "
         "value. Standardize the type before analysis.",
         "Find the values that differ from the dominant type and correct or "
         "convert them.",
         "Medium when a column contains more than one Python value type."),
        (MEDIUM, "Boolean values represented inconsistently",
         inconsistent_bool,
         "These columns appear to hold yes/no or true/false values written "
         "in several different ways.",
         "Standardize the spellings (for example, one form of yes/no), then "
         "convert the columns to a boolean type.",
         "Medium when all values are boolean-like but more than two "
         "spellings exist."),
        (LOW, "Columns that may be boolean", bool_like,
         "These text columns contain only two boolean-like values and may "
         "be easier to analyze as a boolean type.",
         "Consider converting them to a boolean type.",
         "Low when all values are boolean-like with exactly two spellings."),
    ]
    return [make_recommendation(
        priority, "Data Types", issue,
        describe_items([item[1] for item in found]), recommendation, action,
        rule, [item[0] for item in found])
        for priority, issue, found, recommendation, action, rule in specs
        if found]


def analyze_categorical_columns(df):
    """Whitespace, similar names, rare categories and high-cardinality columns."""
    whitespace, similar, rare, cardinality = [], [], [], []
    for column in df.columns:
        series = df[column]
        # Only text/category columns are checked, and text that is really
        # numbers or dates is handled by the data-type analysis.
        if column_kind(series) != "categorical" or \
                pd.api.types.is_bool_dtype(series):
            continue
        if numeric_text_share(series) >= TEXT_CONVERSION_SHARE or \
                date_text_share(series) >= TEXT_CONVERSION_SHARE:
            continue
        values = series.dropna().astype(str)
        if values.empty:
            continue
        stripped = values.str.strip()

        # Leading/trailing spaces split one category into several.
        padded = int((values != stripped).sum())
        if padded:
            whitespace.append((column, f"{column}: {padded:,} values"))

        # Names that match after ignoring case, spaces and punctuation.
        normalized = (stripped.str.lower()
                      .str.replace(r"[\W_]+", "", regex=True))
        pairs = pd.DataFrame({"raw": stripped, "norm": normalized})
        pairs = pairs[pairs["norm"] != ""].drop_duplicates()
        groups = pairs.groupby("norm")["raw"].agg(list)
        groups = groups[groups.map(len) > 1]
        if len(groups):
            example = " / ".join(f'"{name}"' for name in groups.iloc[0][:3])
            similar.append((column, f"{column}: {len(groups):,} group(s), "
                            f"e.g. {example}"))

        # Categories that are very infrequent.
        counts = stripped.value_counts()
        unique_count = len(counts)
        if (len(stripped) >= RARE_MIN_ROWS
                and 3 <= unique_count <= HIGH_CARDINALITY_COUNT):
            rare_count = int((counts / len(stripped)
                              < RARE_CATEGORY_THRESHOLD).sum())
            if rare_count:
                rare.append((column, f"{column}: {rare_count:,} of "
                             f"{unique_count:,} categories"))

        # Too many distinct values for a category chart to be readable.
        ratio = unique_count / len(stripped)
        if unique_count > HIGH_CARDINALITY_COUNT or (
                ratio >= ID_LIKE_RATIO and unique_count > ID_LIKE_MIN_UNIQUE):
            cardinality.append((column, f"{column}: {unique_count:,} distinct "
                                f"values ({ratio:.0%} of rows)"))

    # (priority, issue, found items, recommendation, action, rule)
    specs = [
        (MEDIUM, "Leading or trailing whitespace", whitespace,
         "Values such as \"California\" and \"California \" may represent "
         "the same category. Consider standardizing them.",
         "Review the affected values and trim the extra spaces in a cleaned "
         "copy of the data.",
         "Medium when any text value differs from its trimmed form."),
        (MEDIUM, "Possibly inconsistent category names", similar,
         "Names that differ only by capitalization or punctuation (for "
         "example \"California\" and \"california\") could potentially be "
         "the same category and may split counts.",
         "Review the listed groups and normalize the spellings you confirm "
         "are equivalent.",
         "Medium when two or more raw values are identical after ignoring "
         "case, spaces and punctuation."),
        (LOW, "Rare categories", rare,
         "Some categories are very infrequent and may be typos or may need "
         "grouping before analysis.",
         "Check whether rare categories are valid; consider grouping them "
         "as 'Other' for charts.",
         f"Low when a category is below {RARE_CATEGORY_THRESHOLD:.0%} of "
         f"rows (datasets with >= {RARE_MIN_ROWS} rows)."),
        (LOW, "High number of unique values", cardinality,
         "These columns may be identifiers or free text rather than useful "
         "categories; bar charts and frequency tables will be hard to "
         "interpret.",
         "Consider excluding them from category analysis or grouping their "
         "values.",
         f"Low when a column has more than {HIGH_CARDINALITY_COUNT} distinct "
         f"values, or >= {ID_LIKE_RATIO:.0%} distinct values with more than "
         f"{ID_LIKE_MIN_UNIQUE} of them."),
    ]
    return [make_recommendation(
        priority, "Categorical Data", issue,
        describe_items([item[1] for item in found]), recommendation, action,
        rule, [item[0] for item in found])
        for priority, issue, found, recommendation, action, rule in specs
        if found]


def analyze_numerical_columns(df, numeric_columns):
    """Constant, near-constant, skewed and wide-range numeric columns."""
    constant, near_constant, skewed, wide = [], [], [], []
    for column in numeric_columns:
        values = finite_values(df[column])
        if values.empty:
            continue
        unique_count = values.nunique()
        if unique_count <= 1:
            constant.append((column, str(column)))
            continue
        top_share = values.value_counts(normalize=True).iloc[0]
        if top_share >= NEAR_CONSTANT_SHARE:
            near_constant.append((column, f"{column} ({top_share:.0%} "
                                  "identical)"))
        if len(values) < MIN_ROWS_FOR_STATS:
            continue
        skewness = values.skew()
        if unique_count > 2 and abs(skewness) >= SKEW_THRESHOLD:
            skewed.append((column, f"{column} (skewness {skewness:.2f})"))
        iqr = values.quantile(0.75) - values.quantile(0.25)
        value_range = values.max() - values.min()
        if iqr > 0 and value_range / iqr >= EXTREME_RANGE_RATIO:
            wide.append((column, f"{column} (range is "
                         f"{value_range / iqr:,.0f} x IQR)"))

    # (priority, issue, found items, recommendation, action, rule)
    specs = [
        (MEDIUM, "Constant numerical columns", constant,
         "These columns contain a single value, so they carry no information "
         "for comparing records and may be redundant.",
         "Confirm the value is expected, then consider excluding the columns "
         "from analysis.",
         "Medium when a numeric column has one distinct non-missing value."),
        (LOW, "Very little variation", near_constant,
         "Almost all values in these columns are identical, which limits "
         "what statistics can reveal.",
         "Consider whether the columns are useful for your analysis.",
         f"Low when one value fills >= {NEAR_CONSTANT_SHARE:.0%} of rows."),
        (LOW, "Highly skewed distributions", skewed,
         "Skewed distributions can make the mean unrepresentative; the "
         "median may describe typical values better.",
         "Check the histograms; consider a transformation (such as log) if "
         "your method assumes symmetric data.",
         f"Low when |skewness| >= {SKEW_THRESHOLD} (columns with more than "
         "two distinct values)."),
        (LOW, "Extreme value ranges", wide,
         "The overall range is very large compared with the typical spread, "
         "which may indicate extreme values or mixed units.",
         "Review the minimum and maximum values and confirm the units are "
         "consistent.",
         f"Low when (max - min) / IQR >= {EXTREME_RANGE_RATIO}."),
    ]
    return [make_recommendation(
        priority, "Numerical Analysis", issue,
        describe_items([item[1] for item in found]), recommendation, action,
        rule, [item[0] for item in found])
        for priority, issue, found, recommendation, action, rule in specs
        if found]


def analyze_correlations(df, numeric_columns, analytics_results=None):
    """Strong-association recommendations (association, not causation)."""
    matrix = (analytics_results or {}).get("correlation_matrix")
    source = "the correlation matrix supplied by the analytics section"
    if isinstance(matrix, pd.DataFrame) and not matrix.empty \
            and matrix.shape[0] == matrix.shape[1]:
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
    else:
        # No usable matrix supplied: compute Pearson correlations on columns
        # that vary and have enough observations.
        usable = [c for c in numeric_columns
                  if df[c].nunique(dropna=True) > 1
                  and df[c].notna().sum() >= MIN_ROWS_FOR_STATS]
        if len(usable) < 2:
            return []
        matrix = df[usable].corr()
        source = "Pearson correlations calculated from the dataset"

    # Collect each variable pair once (upper triangle, diagonal excluded).
    coefficients = matrix.to_numpy(dtype=float)
    pairs = []
    for i in range(coefficients.shape[0]):
        for j in range(i + 1, coefficients.shape[1]):
            value = coefficients[i, j]
            if not np.isnan(value) and abs(value) >= STRONG_CORRELATION:
                pairs.append((str(matrix.index[i]), str(matrix.columns[j]),
                              float(value)))
    if not pairs:
        return []
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)

    def label(pair):
        return f"{pair[0]} & {pair[1]} (r = {pair[2]:.2f})"

    very_strong = [p for p in pairs if abs(p[2]) >= REDUNDANT_CORRELATION]
    strong = [p for p in pairs if abs(p[2]) < REDUNDANT_CORRELATION]
    note = ("Correlation shows an association, not causation, and may be "
            "driven by outliers or a small sample.")
    recs = []
    if very_strong:
        recs.append(make_recommendation(
            LOW, "Correlation", "Very strong correlation between variables",
            f"Based on {source}: "
            + describe_items([label(p) for p in very_strong]) + f". {note}",
            "Consider investigating whether these variables measure related "
            "phenomena or contain overlapping information.",
            "Review each pair; if they overlap, consider keeping only one "
            "for modelling.",
            f"Low when |r| >= {REDUNDANT_CORRELATION}.",
            sorted({name for p in very_strong for name in p[:2]})))
    if strong:
        recs.append(make_recommendation(
            INFO, "Correlation", "Strong correlation worth investigating",
            f"Based on {source}: "
            + describe_items([label(p) for p in strong]) + f". {note}",
            "A strong association was detected. Consider investigating the "
            "relationship; do not assume one variable causes the other.",
            "Examine these pairs with the scatter plot analysis.",
            f"Informational when {STRONG_CORRELATION} <= |r| < "
            f"{REDUNDANT_CORRELATION}.",
            sorted({name for p in strong for name in p[:2]})))
    return recs


def analyze_dataset_structure(df, numeric_columns):
    """Size, column mix, date availability and possibly irrelevant columns."""
    rows, columns = df.shape
    recs = []

    # Row-count rule.
    if rows < SMALL_ROWS:
        recs.append(make_recommendation(
            HIGH if rows < VERY_SMALL_ROWS else MEDIUM, "Dataset Structure",
            "Small number of rows", f"The dataset has only {rows:,} rows.",
            "Statistics, correlations and charts based on very few rows may "
            "be unstable. Interpret results with caution.",
            "Collect more observations if possible, or treat findings as "
            "exploratory.",
            f"High when rows < {VERY_SMALL_ROWS}; Medium when rows < "
            f"{SMALL_ROWS}."))

    # Column-count rule.
    if columns > rows or columns > WIDE_DATASET_COLUMNS:
        recs.append(make_recommendation(
            LOW, "Dataset Structure", "Wide dataset",
            f"The dataset has {columns:,} columns and {rows:,} rows.",
            "Having many columns relative to rows can make patterns "
            "unreliable and hard to review.",
            "Consider focusing on the columns most relevant to your "
            "question.",
            f"Low when columns > rows or columns > {WIDE_DATASET_COLUMNS}."))

    # Column-mix rules.
    if not numeric_columns:
        recs.append(make_recommendation(
            LOW, "Dataset Structure", "No numerical columns",
            "No numerical columns were detected.",
            "Numeric statistics, histograms, correlations and scatter plots "
            "will be unavailable. If numbers are stored as text, see the "
            "data-type recommendations.",
            "Verify the data types of the columns.",
            "Low when the numeric column count is zero."))
    if not any(column_kind(df[c]) == "categorical" for c in df.columns):
        recs.append(make_recommendation(
            INFO, "Dataset Structure", "No categorical columns",
            "No text or category columns were detected.",
            "Frequency tables and bar charts will be unavailable.",
            "No action is needed unless categories are expected.",
            "Informational when the categorical column count is zero."))
    if not detect_datetime_columns(df):
        recs.append(make_recommendation(
            INFO, "Dataset Structure", "No date/time column detected",
            "No date or time column was found.",
            "Time-series analysis will be unavailable.",
            "Validate date/time columns if you expect trends over time.",
            "Informational when no date/time column is detected."))

    # Potentially irrelevant columns (constant text, or one value per row).
    irrelevant = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty or column in numeric_columns:
            continue
        try:
            unique_count = series.nunique()
        except TypeError:
            continue
        if unique_count == 1 and len(series) > 1:
            irrelevant.append((column, f"{column} (single value)"))
        elif (unique_count == len(series) and len(series) >= SMALL_ROWS
              and is_text_column(series)):
            irrelevant.append((column, f"{column} (unique per row)"))
    if irrelevant:
        recs.append(make_recommendation(
            LOW, "Dataset Structure", "Potentially irrelevant columns",
            describe_items([item[1] for item in irrelevant]),
            "Constant columns and one-value-per-row text columns "
            "(identifiers, notes) add little to statistical analysis.",
            "Consider whether to exclude these columns from analysis.",
            "Low for non-numeric columns with one value, or with a different "
            f"value in every row (>= {SMALL_ROWS} rows).",
            [item[0] for item in irrelevant]))
    return recs


def run_check(name, analyzer, *args):
    """Run one analyzer; a failure becomes an Informational note, not a crash."""
    try:
        return analyzer(*args)
    except Exception as error:  # one failing check must not stop the others
        return [make_recommendation(
            INFO, "General Recommendations", f"{name} check not completed",
            f"The check could not be performed: {error}",
            "This check was skipped; other results are unaffected.",
            "Review the column data types and re-run the recommendations.",
            "Informational whenever a check cannot be performed.")]


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------
def prioritize_recommendations(recommendations):
    """Return recommendations as a DataFrame sorted by priority then category."""
    table = pd.DataFrame(recommendations, columns=RECOMMENDATION_COLUMNS)
    if table.empty:
        return table
    table["Priority"] = pd.Categorical(table["Priority"],
                                       categories=PRIORITY_ORDER, ordered=True)
    table["Category"] = pd.Categorical(table["Category"],
                                       categories=CATEGORY_ORDER, ordered=True)
    table = table.sort_values(["Priority", "Category"], kind="stable")
    # Plain strings make filtering and CSV export straightforward.
    table["Priority"] = table["Priority"].astype(str)
    table["Category"] = table["Category"].astype(str)
    return table.reset_index(drop=True)


def generate_recommendations(df, quality_results=None, analytics_results=None):
    """Analyze df (read-only) and return the prioritized recommendation table."""
    if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    recommendations = []
    # Duplicate column names are reported, then a renamed COPY is analyzed so
    # the user's original DataFrame is untouched.
    if df.columns.duplicated().any():
        names = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        recommendations.append(make_recommendation(
            MEDIUM, "Dataset Structure", "Duplicate column names",
            f"Repeated column names: {describe_items(names)}.",
            "Duplicate names make column selection ambiguous. Consider "
            "renaming them.",
            "Give each column a unique, descriptive name.",
            "Medium when any column name appears more than once.", names))
    working = df.copy()
    working.columns, _ = make_unique_columns(working.columns)
    numeric_columns = detect_numeric_columns(working)

    # Run every analyzer; each contributes its own data-driven records.
    recommendations += run_check("Missing values", analyze_missing_values,
                                 working)
    recommendations += run_check("Invalid values", analyze_invalid_values,
                                 working, quality_results)
    recommendations += run_check("Duplicates", analyze_duplicates, working)
    recommendations += run_check("Outliers", analyze_outliers, working,
                                 numeric_columns)
    recommendations += run_check("Data types", analyze_data_types, working)
    recommendations += run_check("Categorical data",
                                 analyze_categorical_columns, working)
    recommendations += run_check("Numerical analysis",
                                 analyze_numerical_columns, working,
                                 numeric_columns)
    recommendations += run_check("Correlation", analyze_correlations, working,
                                 numeric_columns, analytics_results)
    recommendations += run_check("Dataset structure",
                                 analyze_dataset_structure, working,
                                 numeric_columns)

    # A general next step appears only when something needs attention.
    if any(r["Priority"] != INFO for r in recommendations):
        recommendations.append(make_recommendation(
            INFO, "General Recommendations", "Re-run analytics after cleaning",
            "One or more potential issues were identified above.",
            "After making any changes, re-run the analytics and these "
            "recommendations to confirm the issues are resolved.",
            "Apply changes to a copy of the data, keep the original, and "
            "re-run the platform.",
            "Informational whenever any High, Medium or Low item exists."))
    return prioritize_recommendations(recommendations)


def build_summary(recommendations):
    """Return (message level, message) generated from the actual findings."""
    counts = recommendations["Priority"].value_counts()
    high, medium, low = (int(counts.get(p, 0)) for p in (HIGH, MEDIUM, LOW))
    scope = ("This assessment reflects only the checks performed: missing "
             "values, placeholders, duplicates, outliers, data types, "
             "categories, numeric distributions, correlations and dataset "
             "structure.")
    if high:
        return "warning", (f"The dataset contains {high} high-priority "
                           f"item(s) that should be reviewed before advanced "
                           f"analysis is performed, plus {medium} medium and "
                           f"{low} low-priority item(s). {scope}")
    if medium or low:
        return "info", (f"The dataset contains {medium} medium and {low} "
                        f"low-priority area(s) that may be worth reviewing. "
                        f"No high-priority items were found. {scope}")
    return "success", ("No major data-quality issues were detected based on "
                       f"the available checks. {scope}")


# ---------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------
def render_cards(table, show_rule):
    """Show each recommendation as an expandable card."""
    for _, row in table.iterrows():
        # Expander title: priority, category and issue, so the user can scan
        # the list and open only the items they care about.
        with st.expander(f"[{row['Priority']}] {row['Category']} - "
                         f"{row['Issue']}"):
            st.markdown(f"**Finding:** {row['Finding']}")
            st.markdown(f"**Recommendation:** {row['Recommendation']}")
            st.markdown(f"**Suggested Action:** {row['Suggested Action']}")
            if row["Affected Columns"]:
                st.markdown(f"**Affected Columns:** {row['Affected Columns']}")
            if show_rule:
                # Shows the exact rule behind the priority for transparency.
                st.caption(f"Priority rule: {row['Rule']}")


def render_grouped(table, groups, show_rule, group_column, headings=None):
    """Render cards under one subheader per group that has rows."""
    shown = False
    for group in groups:
        subset = table[table[group_column] == group]
        if subset.empty:
            continue
        shown = True
        # Subheader separating this group's recommendations from the others.
        st.subheader((headings or {}).get(group, group))
        render_cards(subset, show_rule)
    if not shown:
        # Tell the user the current filters leave nothing to show here.
        st.info("No recommendations in this view match the selected filters.")


# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
def render_recommendations(df, quality_results=None, analytics_results=None):
    """Render the complete Recommendations interface for the given DataFrame."""
    # Main title so the user knows they are viewing recommendations generated
    # from their dataset.
    st.title("Recommendations")

    # Guard: recommendations need a real, non-empty DataFrame.
    if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
        # Explains why nothing else is shown on this page.
        st.warning("No dataset is currently available for recommendations.")
        return

    # Tell the user when earlier-part results were not supplied, so they
    # understand the checks were run directly on the dataset.
    if quality_results is None or analytics_results is None:
        st.info("Data-quality or analytics results were not supplied for "
                "this page, so those checks were run directly on the "
                "dataset.")

    # Generate recommendations from the user's actual DataFrame; the engine
    # only reads it, so the original data is never changed.
    recommendations = generate_recommendations(df, quality_results,
                                               analytics_results)

    # ---------------------------------------------------------
    # OVERALL RECOMMENDATION SUMMARY
    # ---------------------------------------------------------
    # Header separating the overall assessment from individual findings.
    st.header("Overall Recommendation Summary")
    level, message = build_summary(recommendations)
    # Colored banner whose wording and color are determined by the counts of
    # High/Medium/Low items actually found.
    banner = {"warning": st.warning, "info": st.info, "success": st.success}
    banner[level](message)

    # Count tiles for each priority level in the generated table.
    counts = recommendations["Priority"].value_counts()
    tiles = st.columns(5)
    tiles[0].metric("Total Recommendations", len(recommendations))
    for tile, priority in zip(tiles[1:], PRIORITY_ORDER):
        tile.metric(priority, int(counts.get(priority, 0)))

    # Explain exactly how each priority is assigned, using the live threshold
    # constants so the displayed rules always match the code.
    with st.expander("How Priorities Are Assigned"):
        st.markdown(
            f"- **Missing values:** High >= {HIGH_MISSING_THRESHOLD:.0%} of a "
            f"column; Medium >= {MEDIUM_MISSING_THRESHOLD:.0%}; otherwise Low.\n"
            f"- **Invalid/placeholder values:** High >= "
            f"{HIGH_INVALID_THRESHOLD:.0%}; Medium >= "
            f"{MEDIUM_INVALID_THRESHOLD:.0%}; otherwise Low.\n"
            f"- **Duplicates:** High >= {HIGH_DUPLICATE_THRESHOLD:.0%} of "
            f"rows; Medium >= {MEDIUM_DUPLICATE_THRESHOLD:.0%}; otherwise Low.\n"
            f"- **Outliers:** IQR fences ({IQR_MULTIPLIER} x IQR); Medium when "
            f">= {MEDIUM_OUTLIER_THRESHOLD:.0%} of values; otherwise Low. "
            "Never High, because extreme values may be valid.\n"
            "- **Data types, categories, numerics:** Medium when the problem "
            "can distort analysis; Low when it is a minor or optional "
            "improvement.\n"
            f"- **Correlation:** Low when |r| >= {REDUNDANT_CORRELATION}; "
            f"Informational when |r| >= {STRONG_CORRELATION}.\n"
            f"- **Dataset size:** High below {VERY_SMALL_ROWS} rows; Medium "
            f"below {SMALL_ROWS} rows.")

    # Nothing further to filter or list when no checks produced a finding.
    if recommendations.empty:
        return

    # ---------------------------------------------------------
    # FILTER CONTROLS
    # ---------------------------------------------------------
    # Dropdown that limits the displayed recommendations to one priority
    # level; "All" shows every level.
    priority_filter = st.selectbox("Filter Recommendations by Priority",
                                   ["All"] + PRIORITY_ORDER,
                                   key=f"{KEY}_priority")
    # Multiselect that limits the display to chosen recommendation categories;
    # only categories that actually occurred are offered.
    available = [c for c in CATEGORY_ORDER
                 if c in set(recommendations["Category"])]
    category_filter = st.multiselect("Select Recommendation Category",
                                     available, default=available,
                                     key=f"{KEY}_category")
    # Checkbox that adds the exact priority rule to every recommendation, so
    # the user can see why it was rated that way.
    show_rule = st.checkbox("Show Priority Rules", value=False,
                            key=f"{KEY}_show_rule")

    # Apply both filters to the generated recommendations (a new table; the
    # full set is unchanged).
    filtered = recommendations[recommendations["Category"]
                               .isin(category_filter)]
    if priority_filter != "All":
        filtered = filtered[filtered["Priority"] == priority_filter]
    if filtered.empty:
        # Explains that the filters, not the data, caused the empty view.
        st.info("No recommendations match the selected filters.")
        return

    # ---------------------------------------------------------
    # RECOMMENDATION VIEWS
    # ---------------------------------------------------------
    priority_tab, quality_tab, analytics_tab, table_tab = st.tabs(
        ["By Priority", "Data Quality Recommendations",
         "Analytics Recommendations", "Recommendation Table"])
    with priority_tab:
        # Lists High, Medium, Low and Informational items under their own
        # headings so the user can focus on earlier-attention issues first.
        render_grouped(filtered, PRIORITY_ORDER, show_rule, "Priority",
                       PRIORITY_HEADINGS)
    with quality_tab:
        # Groups recommendations about the condition of the raw data.
        render_grouped(filtered, DATA_QUALITY_CATEGORIES, show_rule,
                       "Category")
    with analytics_tab:
        # Groups recommendations about how the data behaves analytically.
        render_grouped(filtered, ANALYTICS_CATEGORIES, show_rule, "Category")
    with table_tab:
        # Structured table of all displayed recommendations.
        st.subheader("Recommendation Table")
        shown_columns = [c for c in RECOMMENDATION_COLUMNS
                         if show_rule or c != "Rule"]
        st.dataframe(filtered[shown_columns], hide_index=True)

    # ---------------------------------------------------------
    # RECOMMENDED NEXT STEPS
    # ---------------------------------------------------------
    # Heading for a short, ordered action list derived from the displayed
    # recommendations (highest priority first, duplicates removed).
    st.header("Recommended Next Steps")
    steps = filtered.drop_duplicates("Suggested Action").head(8)
    for number, (_, row) in enumerate(steps.iterrows(), start=1):
        st.markdown(f"{number}. **{row['Category']}** - "
                    f"{row['Suggested Action']}")

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------
    # Download button so the user can save the displayed recommendations
    # (priority, category, issue, finding, recommendation, action, columns
    # and rule) as a CSV file and use them outside the platform.
    st.download_button(
        label="Download Recommendations",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="recommendations.csv",
        mime="text/csv",
        key=f"{KEY}_download",
    )
