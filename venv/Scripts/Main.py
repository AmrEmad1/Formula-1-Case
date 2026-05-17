# F1 Racing — Exploratory Data Analysis
# Dataset: F1 Races Database (1950–2026)
#
# 7 questions, 7 charts. Each section cleans the data it needs,
# builds the answer step by step, then plots it.


# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
# pandas  → everything table-related (loading, filtering, grouping, merging)
# matplotlib → draws the actual charts
# seaborn → sits on top of matplotlib, makes things look much better by default
# os → just for building file paths that work on any machine
# warnings → silencing a few annoying pandas/matplotlib messages that don't matter

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# Setting the chart style once here so I don't repeat it in every single plot.
# whitegrid = light grid lines in the background, looks clean
sns.set_theme(style="whitegrid", palette="deep")

plt.rcParams.update({
    "figure.dpi"        : 130,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#f8f8f8",
    "axes.spines.top"   : False,       # removing top and right borders
    "axes.spines.right" : False,       # makes charts look less boxy
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
})

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)   # creates the folder if it doesn't exist yet

print("imports done.")


# =============================================================================
# SECTION 2 — LOADING THE DATA
# =============================================================================

DATA_DIR = "data"

results       = pd.read_csv(os.path.join(DATA_DIR, "f1_results.csv"))
drivers       = pd.read_csv(os.path.join(DATA_DIR, "f1_drivers.csv"))
races         = pd.read_csv(os.path.join(DATA_DIR, "f1_races.csv"))
qualifying    = pd.read_csv(os.path.join(DATA_DIR, "f1_qualifying.csv"))
circuits      = pd.read_csv(os.path.join(DATA_DIR, "f1_circuits.csv"))
constructors  = pd.read_csv(os.path.join(DATA_DIR, "f1_constructors.csv"))
con_standings = pd.read_csv(os.path.join(DATA_DIR, "f1_constructor_standings.csv"))

# quick check on what we're working with
file_info = {
    "f1_results"              : results,
    "f1_drivers"              : drivers,
    "f1_races"                : races,
    "f1_qualifying"           : qualifying,
    "f1_circuits"             : circuits,
    "f1_constructors"         : constructors,
    "f1_constructor_standings": con_standings,
}

for name, df in file_info.items():
    print(f"  {name:<30}  {df.shape[0]:>6,} rows  x  {df.shape[1]} cols")


# =============================================================================
# SECTION 3 — MISSING VALUE CHECK
# =============================================================================
# I want to know which columns have gaps before touching anything.
# Some missing values are expected (e.g. Q2 times didn't exist before the modern
# qualifying format), others might be a problem. Better to know upfront.

def show_missing_values(df, name):
    print(f"\n── {name}")
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} cols")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("   no missing values")
    else:
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"   {col:<28}  {count:>5,} missing  ({pct:.1f}%)")


for name, df in file_info.items():
    show_missing_values(df, name)


# =============================================================================
# SECTION 4 — HELPER FUNCTIONS
# =============================================================================

def save_chart(filename):
    """saves the current figure into the charts folder"""
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    print(f"  saved → {path}")


def add_bar_labels_h(ax, color="white", fontsize=9):
    """puts the value number inside each horizontal bar"""
    for bar in ax.patches:
        width = bar.get_width()
        if width > 0:
            ax.text(
                width - 1,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                va="center", ha="right",
                fontsize=fontsize, color=color, fontweight="bold",
            )


def add_bar_labels_v(ax, color="#333333", fontsize=9):
    """puts the value number above each vertical bar"""
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + ax.get_ylim()[1] * 0.01,
                f"{int(height):,}",
                ha="center", va="bottom",
                fontsize=fontsize, color=color, fontweight="bold",
            )


def parse_race_time_to_seconds(time_str):
    """
    Converts a race time string like "1:45:30.123" into total seconds.
    Returns NaN for anything that's a gap time (+1.5s, +1 Lap) or unparseable.
    Only the race winner has an absolute time — everyone else shows a gap.
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return float("nan")

    time_str = time_str.strip()

    # gap times start with "+" or mention "Lap" — skip those
    if time_str.startswith("+") or "Lap" in time_str:
        return float("nan")

    try:
        parts = time_str.split(":")

        if len(parts) == 3:               # H:MM:SS.mmm
            hours   = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        elif len(parts) == 2:             # MM:SS.mmm — rare in older records
            hours   = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        else:
            return float("nan")

        return hours * 3600 + minutes * 60 + seconds

    except (ValueError, IndexError):
        return float("nan")


print("helper functions ready.")


# =============================================================================
# QUESTION 1 — Youngest driver to win the Monaco Grand Prix
# =============================================================================
# Steps:
# 1. filter results for Monaco GP wins (position == 1)
# 2. pull in each winner's date of birth from the drivers table
# 3. pull in the exact race date from the races table
# 4. calculate age at race day in years
# 5. sort ascending — youngest at the top
#
# Rows missing either dob or race_date get dropped. Can't calculate age without both.
# Showing top 5 in the chart for context, not just the winner.

print("\n" + "=" * 55)
print("Q1 — Youngest Monaco Grand Prix Winner")
print("=" * 55)

monaco_winners = results[
    (results["race_name"].str.contains("Monaco Grand Prix", na=False)) &
    (results["position"] == 1)
].copy()

print(f"\n  Monaco GP races in dataset: {monaco_winners['season'].nunique()}")

# merge in date of birth
monaco_winners = monaco_winners.merge(
    drivers[["driver_id", "dob"]],
    on="driver_id", how="left"
)

# merge in race date — need both season and round to uniquely identify a race
monaco_winners = monaco_winners.merge(
    races[["season", "round", "date"]].rename(columns={"date": "race_date"}),
    on=["season", "round"], how="left"
)

# convert strings to actual dates — errors='coerce' turns bad values into NaT
monaco_winners["dob"]       = pd.to_datetime(monaco_winners["dob"],       errors="coerce")
monaco_winners["race_date"] = pd.to_datetime(monaco_winners["race_date"], errors="coerce")
monaco_winners              = monaco_winners.dropna(subset=["dob", "race_date"])

# age in years — dividing by 365.25 to account for leap years
monaco_winners["age_at_race"] = (
    (monaco_winners["race_date"] - monaco_winners["dob"]).dt.days / 365.25
)

monaco_winners_sorted = monaco_winners.sort_values("age_at_race")
youngest = monaco_winners_sorted.iloc[0]

age_years  = int(youngest["age_at_race"])
age_months = int((youngest["age_at_race"] - age_years) * 12)

print(f"\n  Driver    : {youngest['driver_name']}")
print(f"  Year      : {int(youngest['season'])}")
print(f"  Age       : {age_years} yrs, {age_months} months")
print(f"  Born      : {youngest['dob'].strftime('%d %b %Y')}")
print(f"  Race date : {youngest['race_date'].strftime('%d %b %Y')}")

# chart — top 5 youngest
top5 = monaco_winners_sorted.head(5).copy()
top5["label"] = top5["driver_name"] + "  (" + top5["season"].astype(int).astype(str) + ")"

fig, ax = plt.subplots(figsize=(9, 4))
colours = sns.color_palette("Reds_r", n_colors=5)
bars    = ax.barh(top5["label"], top5["age_at_race"], color=colours, edgecolor="white", height=0.55)

for bar, age in zip(bars, top5["age_at_race"]):
    yrs = int(age)
    mth = int((age - yrs) * 12)
    ax.text(
        bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
        f"{yrs}y {mth}m",
        va="center", ha="right", fontsize=9, color="white", fontweight="bold"
    )

ax.set_xlabel("Age at Race Day (years)")
ax.set_title("Top 5 Youngest Monaco Grand Prix Winners")
ax.invert_yaxis()
ax.set_xlim(right=top5["age_at_race"].max() * 1.06)

plt.tight_layout()
save_chart("q1_youngest_monaco_winner.png")
plt.show()


# =============================================================================
# QUESTION 2 — Top 10 constructors by total race wins
# =============================================================================
# The constructor standings table has a "wins" column per season per team.
# Grouping by constructor name and summing gives total all-time wins.
# Simple enough — no merges needed here.

print("\n" + "=" * 55)
print("Q2 — Top 10 Constructors by Race Wins")
print("=" * 55)

constructor_wins = (
    con_standings
    .groupby("constructor", as_index=False)["wins"]
    .sum()
    .sort_values("wins", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

season_min = con_standings["season"].min()
season_max = con_standings["season"].max()

print(f"\n  Data range: {season_min}–{season_max}")
print(constructor_wins.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    constructor_wins["constructor"],
    constructor_wins["wins"],
    color=sns.color_palette("tab10", n_colors=10),
    edgecolor="white", height=0.6,
)

add_bar_labels_h(ax, color="white", fontsize=10)

ax.set_xlabel("Total Race Wins")
ax.set_title(f"Top 10 F1 Constructors by Total Race Wins  ({season_min}–{season_max})")
ax.invert_yaxis()
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.set_xlim(right=constructor_wins["wins"].max() * 1.1)

plt.tight_layout()
save_chart("q2_top10_constructor_wins.png")
plt.show()


# =============================================================================
# QUESTION 3 — Q2 qualifying appearances by driver nationality
# =============================================================================
# Q2 only exists in the modern 3-part qualifying format, so older seasons
# will have nulls in that column — that's expected, not missing data.
# A non-null q2 time = driver made it through Q1 into Q2.
# Counting those rows per nationality gives us who's most represented.

print("\n" + "=" * 55)
print("Q3 — Q2 Appearances by Driver Nationality")
print("=" * 55)

q2_data = qualifying[qualifying["q2"].notna()].copy()

print(f"\n  Q2 records: {len(q2_data):,}")
print(f"  Seasons:    {q2_data['season'].min()}–{q2_data['season'].max()}")

q2_with_nationality = q2_data.merge(
    drivers[["driver_id", "nationality"]],
    on="driver_id", how="left"
).dropna(subset=["nationality"])

q2_by_country = (
    q2_with_nationality
    .groupby("nationality", as_index=False)
    .size()
    .rename(columns={"size": "q2_appearances"})
    .sort_values("q2_appearances", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

print(q2_by_country.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(
    q2_by_country["nationality"],
    q2_by_country["q2_appearances"],
    color=sns.color_palette("viridis", n_colors=10),
    edgecolor="white", width=0.6,
)

add_bar_labels_v(ax, fontsize=9)

ax.set_xlabel("Driver Nationality")
ax.set_ylabel("Q2 Appearances")
ax.set_title(
    f"Top 10 Countries by Driver Q2 Appearances\n"
    f"(Modern qualifying: {q2_data['season'].min()}–{q2_data['season'].max()})"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
save_chart("q3_q2_concentration_by_country.png")
plt.show()


# =============================================================================
# QUESTION 4 — Top 10 longest F1 careers
# =============================================================================
# Career = number of seasons from first race to last race (+1 because both ends count).
# Measured in seasons, not races — cleaner and easier to explain.

print("\n" + "=" * 55)
print("Q4 — Top 10 Longest F1 Careers")
print("=" * 55)

career_span = (
    results
    .groupby("driver_name")["season"]
    .agg(first_season="min", last_season="max")
    .reset_index()
)

career_span["career_years"] = career_span["last_season"] - career_span["first_season"] + 1

top10_careers = (
    career_span
    .sort_values("career_years", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

print(top10_careers[["driver_name", "first_season", "last_season", "career_years"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(
    top10_careers["driver_name"],
    top10_careers["career_years"],
    color=sns.color_palette("rocket_r", n_colors=10),
    edgecolor="white", height=0.6,
)

for bar, (_, row) in zip(bars, top10_careers.iterrows()):
    ax.text(
        bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
        f"{int(row['first_season'])}–{int(row['last_season'])}  ({int(row['career_years'])} yrs)",
        va="center", ha="right", fontsize=8.5, color="white", fontweight="bold"
    )

ax.set_xlabel("Career Duration (Seasons)")
ax.set_title("Top 10 F1 Drivers with the Longest Careers")
ax.invert_yaxis()
ax.set_xlim(right=top10_careers["career_years"].max() * 1.08)

plt.tight_layout()
save_chart("q4_longest_careers.png")
plt.show()


# =============================================================================
# QUESTION 5 — Monaco GP winning time trend (1950–present)
# =============================================================================
# Only the race winner has an absolute finishing time.
# Everyone else's time is shown as a gap to the winner ("+1.5s", "+1 Lap", etc.)
# so the parser skips those and only keeps full "H:MM:SS" format strings.
#
# Adding a 5-race rolling average as a second line to show the overall trend
# without the noise of individual outlier years (safety car races, rain, etc.)

print("\n" + "=" * 55)
print("Q5 — Monaco GP Winning Time Trend")
print("=" * 55)

monaco_times = results[
    (results["race_name"].str.contains("Monaco Grand Prix", na=False)) &
    (results["position"] == 1)
].copy()

monaco_times["time_seconds"] = monaco_times["time"].apply(parse_race_time_to_seconds)
monaco_times = monaco_times.dropna(subset=["time_seconds"])
monaco_times["time_minutes"] = monaco_times["time_seconds"] / 60
monaco_times = monaco_times.sort_values("season").reset_index(drop=True)

print(f"\n  Races with valid times: {len(monaco_times)}")
print(f"  Season range: {int(monaco_times['season'].min())}–{int(monaco_times['season'].max())}")

fastest_idx = monaco_times["time_minutes"].idxmin()
slowest_idx = monaco_times["time_minutes"].idxmax()
print(f"  Fastest: {int(monaco_times.loc[fastest_idx, 'season'])} — {monaco_times.loc[fastest_idx, 'time_minutes']:.1f} min")
print(f"  Slowest: {int(monaco_times.loc[slowest_idx, 'season'])} — {monaco_times.loc[slowest_idx, 'time_minutes']:.1f} min")

# rolling average — min_periods=1 so the first few data points still get a value
rolling_avg = (
    monaco_times.set_index("season")["time_minutes"]
    .rolling(window=5, min_periods=1)
    .mean()
)

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    monaco_times["season"], monaco_times["time_minutes"],
    color="#e63946", linewidth=2,
    marker="o", markersize=4, markerfacecolor="white", markeredgewidth=1.5,
    label="Winning time (minutes)", zorder=3,
)

ax.plot(
    rolling_avg.index, rolling_avg.values,
    color="#457b9d", linewidth=2.5, linestyle="--", alpha=0.8,
    label="5-race rolling average", zorder=2,
)

# annotate fastest
fastest_row = monaco_times.loc[fastest_idx]
ax.annotate(
    f"Fastest:\n{int(fastest_row['season'])}\n{fastest_row['time_minutes']:.0f} min",
    xy=(fastest_row["season"], fastest_row["time_minutes"]),
    xytext=(fastest_row["season"] - 10, fastest_row["time_minutes"] + 15),
    fontsize=8, color="#333333",
    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
)

# annotate slowest
slowest_row = monaco_times.loc[slowest_idx]
ax.annotate(
    f"Slowest:\n{int(slowest_row['season'])}\n{slowest_row['time_minutes']:.0f} min",
    xy=(slowest_row["season"], slowest_row["time_minutes"]),
    xytext=(slowest_row["season"] + 5, slowest_row["time_minutes"] - 25),
    fontsize=8, color="#333333",
    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
)

ax.set_xlabel("Season")
ax.set_ylabel("Winning Time (minutes)")
ax.set_title("Monaco Grand Prix Winning Time Trend (1950–present)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)} min"))

plt.tight_layout()
save_chart("q5_monaco_finishing_time_trend.png")
plt.show()


# =============================================================================
# QUESTION 6 — Most common DNF reasons
# =============================================================================
# DNF = Did Not Finish. Filtering out anyone who just finished slowly:
#   - "Finished" → normal completion, exclude
#   - "+X Lap(s)" → completed the race but got lapped, exclude
#   - "Lapped" and "Not classified" → same idea, exclude
#
# Everything left is a genuine failure or incident.
# Keeping "Retired" (no specific reason recorded) and "Disqualified".

print("\n" + "=" * 55)
print("Q6 — DNF Reasons")
print("=" * 55)

dnf_data = results[results["status"] != "Finished"].copy()
dnf_data = dnf_data[~dnf_data["status"].str.match(r"^\+\d+ Laps?$", na=False)]
dnf_data = dnf_data[~dnf_data["status"].isin(["Lapped", "Not classified"])]

dnf_counts = (
    dnf_data["status"]
    .value_counts()
    .reset_index()
    .head(20)
)
dnf_counts.columns = ["status", "occurrences"]

print(f"\n  DNF records (after filtering): {len(dnf_data):,}")
print(dnf_counts.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(
    dnf_counts["status"], dnf_counts["occurrences"],
    color=sns.color_palette("flare", n_colors=len(dnf_counts)),
    edgecolor="white", height=0.65,
)

# labels outside the bars here — some bars are too thin to fit text inside
for bar in ax.patches:
    ax.text(
        bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
        f"{int(bar.get_width()):,}",
        va="center", ha="left", fontsize=8.5, color="#333333"
    )

ax.set_xlabel("Number of Occurrences")
ax.set_title("Top 20 Reasons for Not Finishing (DNF)\nExcluding lapped / classified finishers")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(right=dnf_counts["occurrences"].max() * 1.18)

plt.tight_layout()
save_chart("q6_dnf_reasons.png")
plt.show()


# =============================================================================
# QUESTION 7 — Races with the most accident incidents
# =============================================================================
# Grouping by season + race name because the same race (e.g. British GP) happens
# every year — need both fields to identify a specific event.
# Including all four accident-related statuses, not just "Accident".

print("\n" + "=" * 55)
print("Q7 — Most Accident-Prone Races")
print("=" * 55)

accident_statuses = ["Accident", "Collision", "Collision damage", "Fatal accident"]
accident_data     = results[results["status"].isin(accident_statuses)].copy()

print(f"\n  Total accident records: {len(accident_data):,}")
print(accident_data["status"].value_counts().to_string())

accidents_per_race = (
    accident_data
    .groupby(["season", "race_name"], as_index=False)
    .size()
    .rename(columns={"size": "accident_count"})
    .sort_values("accident_count", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# two-line label so the chart doesn't get cramped
accidents_per_race["race_label"] = (
    accidents_per_race["race_name"] + "\n("
    + accidents_per_race["season"].astype(int).astype(str) + ")"
)

print(accidents_per_race[["season", "race_name", "accident_count"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    accidents_per_race["race_label"], accidents_per_race["accident_count"],
    color=sns.color_palette("Reds_r", n_colors=10),
    edgecolor="white", height=0.6,
)

for bar in ax.patches:
    ax.text(
        bar.get_width() - 0.2, bar.get_y() + bar.get_height() / 2,
        f"{int(bar.get_width())}",
        va="center", ha="right", fontsize=11, color="white", fontweight="bold"
    )

ax.set_xlabel("Number of Accidents / Collisions")
ax.set_title(
    "Top 10 F1 Races by Accident Count\n"
    "(Accident, Collision, Collision Damage, Fatal Accident)"
)
ax.invert_yaxis()
ax.set_xlim(right=accidents_per_race["accident_count"].max() * 1.12)

plt.tight_layout()
save_chart("q7_top10_accident_races.png")
plt.show()


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 55)
print("done — all charts saved to charts/")
print("=" * 55)

summary = [
    ("Q1", "Youngest Monaco winner",   "Lewis Hamilton — 23y 4m (2008)"),
    ("Q2", "Top constructor by wins",  "Ferrari — 223 wins"),
    ("Q3", "Top Q2 nationality",       "Germany — 1,006 appearances"),
    ("Q4", "Longest career",           "Fernando Alonso — 26 seasons"),
    ("Q5", "Fastest Monaco race",      "1984 — 61 min"),
    ("Q6", "Top DNF reason",           "Engine — 2,011 retirements"),
    ("Q7", "Most accidents in one GP", "1975 British GP — 16 incidents"),
]

print()
for q, label, finding in summary:
    print(f"  [{q}]  {label:<30}  →  {finding}")
