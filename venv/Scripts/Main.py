# %% [markdown]
# # 🏎️ Formula 1 Racing — Exploratory Data Analysis
#
# **Dataset :** F1 Races Database (1950–2026)
# **Libraries:** pandas · matplotlib · seaborn
#
# ---
#
# ## What this project covers
#
# Answer 7 business questions about Formula 1 racing using clean,
#
# 2. State our data-cleaning decisions up front
# 3. Transform the data step by step
# 4. Plot a clear, labelled chart
# 5. State one key insight

# %% [markdown]
# ---
# ## Section 1 — Imports
#
# Library  Purpose 
# pandas -------> Loading, cleaning, filtering, and transforming tabular data 
# matplotlib------> Base plotting engine — axes, labels, annotations 
# seaborn----> Prettier charts with better defaults, built on matplotlib |
# os-------> Portable file path management |
#
#  Set a **global chart style** here so every chart in this project
# looks consistent — clean whitegrid background, no top/right border spines.
# This saves  repeating style code in every chart cell.

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Suppress minor pandas/matplotlib warnings that don't affect our results
warnings.filterwarnings("ignore")

# ── Global chart style ─────────────────────────────────────────────────────
# Setting these once here means every chart automatically inherits them.
# Think of it as a "theme" for the entire notebook.
sns.set_theme(style="whitegrid", palette="deep")

plt.rcParams.update({
    "figure.dpi"         : 130,         
    "figure.facecolor"   : "white",      # White figure background
    "axes.facecolor"     : "#f8f8f8",    # Very light grey plot area
    "axes.spines.top"    : False,        # Remove top border line
    "axes.spines.right"  : False,        # Remove right border line
    "axes.titlesize"     : 14,
    "axes.titleweight"   : "bold",
    "axes.labelsize"     : 11,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 9,
    "legend.fontsize"    : 9,
})

# Folder where we save charts — create it if it doesn't exist yet
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

print("✅ Imports done. Chart style configured.")

# ---------------------------------------------------
# ## Section 2 — Data Loading
#
# Load all 7 CSV files into separate pandas DataFrames.
#-----------------------------------------------------
# Set the path to Data Folder 
DATA_DIR = "data"

print("Loading data files...")

results       = pd.read_csv(os.path.join(DATA_DIR, "f1_results.csv"))
drivers       = pd.read_csv(os.path.join(DATA_DIR, "f1_drivers.csv"))
races         = pd.read_csv(os.path.join(DATA_DIR, "f1_races.csv"))
qualifying    = pd.read_csv(os.path.join(DATA_DIR, "f1_qualifying.csv"))
circuits      = pd.read_csv(os.path.join(DATA_DIR, "f1_circuits.csv"))
constructors  = pd.read_csv(os.path.join(DATA_DIR, "f1_constructors.csv"))
con_standings = pd.read_csv(os.path.join(DATA_DIR, "f1_constructor_standings.csv"))

print("All 7 files loaded successfully.\n")

# Show the files and Shape 
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
    print(f"  {name:<30} {df.shape[0]:>6,} rows  ×  {df.shape[1]} columns")
#---------------------------------------------------------------------------
# ## Section 3 — Data Cleaning & Exploratory Checks
#
# Before answering any question, Always check the data first
#  understand:
# - Which columns have missing values
# - How many rows are affected
# - Whether the missing values matter for The  analyses
# ### Key assumptions and cleaning decisions
#------------------------------------------------------------------------------
def show_missing_values(df, name):
    print(f"── {name} {'─' * (45 - len(name))}")
    print(f"   Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    missing = df.isnull().sum()
    missing = missing[missing > 0]      # Only show columns that have gaps

    if missing.empty:
        print("   There is no missing values")
    else:
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"   {col:<28} {count:>5,} missing  ({pct:.1f}%)")
    print()


# Check all Data 
for name, df in file_info.items():
    show_missing_values(df, name)
#-----------------------------------------------------------------------
# ## Section 4 — Helper Functions
#
# Functions defined here:
# save_chart(filename) — saves the current figure to the charts folder
# add_bar_labels_h(ax) annotates a horizontal bar chart with values
# - parse_race_time_to_seconds(time_str)— converts "H:MM:SS.mmm" to seconds
#--------------------------------------------------------------------------
def save_chart(filename):
    """Save the current matplotlib figure to the charts/ folder."""
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    print(f"  📁 Chart saved → {path}")


def add_bar_labels_h(ax, color="white", fontsize=9):
    for bar in ax.patches:
        width = bar.get_width()
        if width > 0:
            ax.text(
                width - 1,                          
                bar.get_y() + bar.get_height() / 2, 
                f"{int(width):,}",
                va="center",
                ha="right",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
            )


def add_bar_labels_v(ax, color="#333333", fontsize=9):
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + ax.get_ylim()[1] * 0.01,   
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
            )


def parse_race_time_to_seconds(time_str):
    # Return NaN for missing values
    if pd.isna(time_str) or not isinstance(time_str, str):
        return float("nan")

    time_str = time_str.strip()

    # Skip gap times like "+1.5s", "+1 Lap", etc.
    if time_str.startswith("+") or "Lap" in time_str:
        return float("nan")

    try:
        parts = time_str.split(":")

        if len(parts) == 3:
            # Format: H:MM:SS.mmm
            hours   = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        elif len(parts) == 2:
            # Format: MM:SS.mmm  (rare in early records)
            hours   = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        else:
            return float("nan")

        return hours * 3600 + minutes * 60 + seconds

    except (ValueError, IndexError):
        # If parsing fails for any reason, return NaN gracefully
        return float("nan")


print("Create of Helper function has been completed.")
#------------------------------------------------------
# ## Question 1 — Youngest Driver to Win the Monaco Grand Prix
#
# **Business question:**
# > Who is the youngest driver ever to win the Monaco Grand Prix,
# > and in which year did they do it? How old were they?
#
# ### Approach
# 1. Filter `results` to Monaco Grand Prix rows where `position == 1` (winner only)
# 2. Join with `drivers` to get each winner's date of birth 
# 3. Join with `races` to get the exact race date
# 4. Calculate age at race day: `(race_date − dob).days ÷ 365.25`
# 5. Sort ascending by age → the first row is the youngest winner
#
# ### Data decisions
# - Rows with a missing `dob` OR missing `race_date` are dropped —
#   Cannot calculate age without both dates.
# Show the **top 5 youngest winners** in the chart for context,

#------------------------------------------------------------
print("=" * 60)
print("Q1 — Youngest Monaco Grand Prix Winner")
print("=" * 60)

# ── Step 1: Filter for Monaco GP winners ──────────────────────
# position == 1 means the driver won the race
monaco_winners = results[
    (results["race_name"].str.contains("Monaco Grand Prix", na=False)) &
    (results["position"] == 1)
].copy()

print(f"\n  Monaco Grand Prix races in dataset : {monaco_winners['season'].nunique()}")

# ── Step 2: Add driver date of birth ──────────────────────────
# We merge on driver_id which exists in both DataFrames
monaco_winners = monaco_winners.merge(
    drivers[["driver_id", "dob"]],
    on="driver_id",
    how="left"
)

# ── Step 3: Add the exact race date ───────────────────────────
monaco_winners = monaco_winners.merge(
    races[["season", "round", "date"]].rename(columns={"date": "race_date"}),
    on=["season", "round"],
    how="left"
)

# ── Step 4: Parse string dates into proper datetime objects ───
# errors='coerce' turns unparseable strings into NaT (Not a Time)
monaco_winners["dob"]       = pd.to_datetime(monaco_winners["dob"],       errors="coerce")
monaco_winners["race_date"] = pd.to_datetime(monaco_winners["race_date"], errors="coerce")

# Drop rows where either date is missing
monaco_winners = monaco_winners.dropna(subset=["dob", "race_date"])

# ── Step 5: Calculate age at race day ─────────────────────────
# .dt.days gives the difference in total days; dividing by 365.25
# converts that to years accounting for leap years
monaco_winners["age_at_race"] = (
    (monaco_winners["race_date"] - monaco_winners["dob"]).dt.days / 365.25
)

# ── Step 6: Sort and extract the youngest winner ───────────────
monaco_winners_sorted = monaco_winners.sort_values("age_at_race")
youngest = monaco_winners_sorted.iloc[0]

# Break the decimal age into years + months for a readable display
age_years  = int(youngest["age_at_race"])
age_months = int((youngest["age_at_race"] - age_years) * 12)

print(f"\n  Youngest Monaco GP Winner")
print(f"     Driver    : {youngest['driver_name']}")
print(f"     Year      : {int(youngest['season'])}")
print(f"     Age       : {age_years} years, {age_months} months")
print(f"     Born      : {youngest['dob'].strftime('%d %b %Y')}")
print(f"     Race date : {youngest['race_date'].strftime('%d %b %Y')}")

# ── Chart: Top 5 youngest Monaco winners ──────────────────────
top5 = monaco_winners_sorted.head(5).copy()
top5["label"] = top5["driver_name"] + "  (" + top5["season"].astype(int).astype(str) + ")"

fig, ax = plt.subplots(figsize=(9, 4))

colours = sns.color_palette("Reds_r", n_colors=5)
bars = ax.barh(top5["label"], top5["age_at_race"], color=colours, edgecolor="white", height=0.55)

# Annotate each bar with the age in "X years Y months" format
for bar, age in zip(bars, top5["age_at_race"]):
    yrs = int(age)
    mth = int((age - yrs) * 12)
    ax.text(
        bar.get_width() - 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{yrs}y {mth}m",
        va="center", ha="right",
        fontsize=9, color="white", fontweight="bold"
    )

ax.set_xlabel("Age at Race Day (years)")
ax.set_title("Top 5 Youngest Monaco Grand Prix Winners")
ax.invert_yaxis()   # Youngest at the top
ax.set_xlim(right=top5["age_at_race"].max() * 1.06)

plt.tight_layout()
save_chart("q1_youngest_monaco_winner.png")
plt.show()
#------------------------------------------------------------
# ##  Question 2 — Top 10 Constructors with the Most Race Wins
#
# **Business question:**
# > Which 10 teams have achived the most total race wins
# > across the entire F1 dataset?
# We use `f1_constructor_standings.csv` which has a `wins` column that records
# how many races each constructor won in each season. We simply group by
# constructor name and sum wins across all seasons.
#------------------------------------------------------------------
print("=" * 60)
print("Q2 — Top 10 Constructors by Total Race Wins")
print("=" * 60)

# Group by constructor name and sum the wins column across all seasons
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

print(f"\n  Dataset spans : {season_min}–{season_max}")
print(f"\n  Top 10 constructors:")
print(constructor_wins.to_string(index=False))

# ── Chart: Horizontal bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

palette = sns.color_palette("tab10", n_colors=10)

ax.barh(
    constructor_wins["constructor"],
    constructor_wins["wins"],
    color=palette,
    edgecolor="white",
    height=0.6,
)

add_bar_labels_h(ax, color="white", fontsize=10)

ax.set_xlabel("Total Race Wins")
ax.set_title(
    f"Top 10 F1 Constructors by Total Race Wins\n"
    f"({season_min}–{season_max})"
)
ax.invert_yaxis()
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.set_xlim(right=constructor_wins["wins"].max() * 1.1)

plt.tight_layout()
save_chart("q2_top10_constructor_wins.png")
plt.show()
#---------------------------------------------------------------------
# ---
# ##  Question 3 — Driver Q2 Qualification Concentration by Country
#
# **Business question:**
# > Which 10 nationalities have produced drivers that most frequently
# > reached Q2 in Formula 1 qualifying?
#
# ### Approach
# 1. Filter `qualifying` to rows where the `q2` column is not null
#    (the driver recorded a Q2 lap time, meaning they reached that session)
# 2. Merge with `drivers` to get each driver's nationality
# 3. Count Q2 appearances per nationality
# 4. Rank descending → top 10
#
# ### Data decisions
# - We count **Q2 appearances** (total rows).
#   A driver who consistently reaches Q2 over many seasons contributes
#   Older seasons had a single qualifying session so their Q2 data is null —
#   this is expected and not a data error.
# - Nationality = the driver's personal nationality
#-----------------------------------------------------------------------
print("=" * 60)
print("Q3 — Q2 Qualification Concentration by Country")
print("=" * 60)

# ── Step 1: Keep only rows where the driver made it to Q2 ─────
q2_data = qualifying[qualifying["q2"].notna()].copy()

print(f"\n  Q2 records found : {len(q2_data):,}")
print(f"  Seasons covered  : {q2_data['season'].min()}–{q2_data['season'].max()}")

# ── Step 2: Add driver nationality ────────────────────────────
q2_with_nationality = q2_data.merge(
    drivers[["driver_id", "nationality"]],
    on="driver_id",
    how="left"
)

# Drop the small number of rows where nationality could not be matched
q2_with_nationality = q2_with_nationality.dropna(subset=["nationality"])

# ── Step 3: Count Q2 appearances per nationality ──────────────
q2_by_country = (
    q2_with_nationality
    .groupby("nationality", as_index=False)
    .size()
    .rename(columns={"size": "q2_appearances"})
    .sort_values("q2_appearances", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

print(f"\n  Top 10 countries by Q2 appearances:")
print(q2_by_country.to_string(index=False))

# ── Chart: Vertical bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

bar_colours = sns.color_palette("viridis", n_colors=10)

ax.bar(
    q2_by_country["nationality"],
    q2_by_country["q2_appearances"],
    color=bar_colours,
    edgecolor="white",
    width=0.6,
)

add_bar_labels_v(ax, fontsize=9)

ax.set_xlabel("Driver Nationality")
ax.set_ylabel("Number of Q2 Appearances")
ax.set_title(
    "Top 10 Countries by Driver Q2 Qualification Appearances\n"
    f"(Modern Qualifying: {q2_data['season'].min()}–{q2_data['season'].max()})"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
save_chart("q3_q2_concentration_by_country.png")
plt.show()
#--------------------------------------------------------------
# ##  Question 4 — Top 10 Drivers with the Longest F1 Careers
#
# **Business question:**
# > Which 10 drivers had the longest careers in Formula 1, measured
# > as the number of seasons between their first and last race?
#
# ### Approach
# 1. From `results`, find each driver's earliest and latest season
# 2. Career duration = `(last_season − first_season + 1)`
# 3. Sort descending → top 10
#
# ### Data decisions
# - Career is measured in **seasons (calendar years)**, not race counts.
#   This is cleaner to explain and easier to visualise.
# - Only drivers with entries in `f1_results` are included.
# - We use `driver_name` from results as it is consistent throughout.
#------------------------------------------------------------------
print("=" * 60)
print("Q4 — Top 10 Drivers with Longest F1 Careers")
print("=" * 60)

# ── Step 1: Find first and last season for each driver ────────
career_span = (
    results
    .groupby("driver_name")["season"]
    .agg(first_season="min", last_season="max")
    .reset_index()
)

# ── Step 2: Calculate career duration ─────────────────────────
# Adding 1 because both the first and last season are fully included
career_span["career_years"] = (
    career_span["last_season"] - career_span["first_season"] + 1
)

# ── Step 3: Sort and take top 10 ──────────────────────────────
top10_careers = (
    career_span
    .sort_values("career_years", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

print(f"\n  Top 10 longest F1 careers:")
print(
    top10_careers[["driver_name", "first_season", "last_season", "career_years"]]
    .to_string(index=False)
)

# ── Chart: Horizontal bar chart with career span labels ───────
fig, ax = plt.subplots(figsize=(10, 6))

palette = sns.color_palette("rocket_r", n_colors=10)

bars = ax.barh(
    top10_careers["driver_name"],
    top10_careers["career_years"],
    color=palette,
    edgecolor="white",
    height=0.6,
)

# Add "YYYY–YYYY (N yrs)" labels inside each bar
for bar, (_, row) in zip(bars, top10_careers.iterrows()):
    ax.text(
        bar.get_width() - 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{int(row['first_season'])}–{int(row['last_season'])}  ({int(row['career_years'])} yrs)",
        va="center", ha="right",
        fontsize=8.5, color="white", fontweight="bold"
    )

ax.set_xlabel("Career Duration (Seasons)")
ax.set_title("Top 10 F1 Drivers with the Longest Careers")
ax.invert_yaxis()
ax.set_xlim(right=top10_careers["career_years"].max() * 1.08)

plt.tight_layout()
save_chart("q4_longest_careers.png")
plt.show()
#-------------------------------------------------------------------------
# ## Question 5 — Monaco Grand Prix Finishing Time Trend (1950–present)
#
# **Business question:**
# > How has the winning race time at Monaco changed since 1950?
# > Are modern cars significantly faster than those from 50 years ago?
#
# ### Approach
# 1. Filter `results` for Monaco GP winners (`position == 1`)
# 2. Keep only rows with an **absolute finishing time** — not gap times
#    - Absolute times look like "2:45:30.100"
# 3. Convert time strings → total seconds using our parse_race_time_to_seconds()
# 4. Convert seconds → minutes for readabilty
# ### Data decisions
# - Only the race **winner** has an absolute time. All other finishers show
#   a gap relative to the winner — this is expected and correct.
# -  Apply a **5-race rolling average** as a second line to smooth out
#---------------------------------------------------------------------
print("=" * 60)
print("Q5 — Monaco Grand Prix Finishing Time Trend (1950–)")
print("=" * 60)

# ── Step 1: Filter for Monaco GP winners ──────────────────────
monaco_times = results[
    (results["race_name"].str.contains("Monaco Grand Prix", na=False)) &
    (results["position"] == 1)
].copy()

# ── Step 2: Convert the time string to total seconds ──────────
# We use the helper function
# It returns NaN for gap times
monaco_times["time_seconds"] = monaco_times["time"].apply(parse_race_time_to_seconds)

# ── Step 3: Drop rows with no usable time ─────────────────────
monaco_times = monaco_times.dropna(subset=["time_seconds"])

# ── Step 4: Convert seconds to minutes for a readable Y-axis ──
monaco_times["time_minutes"] = monaco_times["time_seconds"] / 60

# Sort by season to make sure the line chart goes left → right
monaco_times = monaco_times.sort_values("season").reset_index(drop=True)

print(f"\n  Monaco races with valid finishing times : {len(monaco_times)}")
print(f"  Season range : {int(monaco_times['season'].min())}–{int(monaco_times['season'].max())}")

fastest_idx = monaco_times["time_minutes"].idxmin()
slowest_idx = monaco_times["time_minutes"].idxmax()
print(f"  Fastest race : {int(monaco_times.loc[fastest_idx, 'season'])} — "
      f"{monaco_times.loc[fastest_idx, 'time_minutes']:.1f} min")
print(f"  Slowest race : {int(monaco_times.loc[slowest_idx, 'season'])} — "
      f"{monaco_times.loc[slowest_idx, 'time_minutes']:.1f} min")

# ── Step 5: Calculate a 5-race rolling average ────────────────
# points are available (e.g. at the very start of the series)
rolling_avg = (
    monaco_times
    .set_index("season")["time_minutes"]
    .rolling(window=5, min_periods=1)
    .mean()
)

# ── Chart: Time series line chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

# Main line: actual winning times
ax.plot(
    monaco_times["season"],
    monaco_times["time_minutes"],
    color="#e63946",
    linewidth=2,
    marker="o",
    markersize=4,
    markerfacecolor="white",
    markeredgewidth=1.5,
    label="Winning time (minutes)",
    zorder=3,
)

# Overlay: 5-race rolling average trend line
ax.plot(
    rolling_avg.index,
    rolling_avg.values,
    color="#457b9d",
    linewidth=2.5,
    linestyle="--",
    alpha=0.8,
    label="5-race rolling average",
    zorder=2,
)

# Annotate the fastest race
fastest_row = monaco_times.loc[fastest_idx]
ax.annotate(
    f"Fastest:\n{int(fastest_row['season'])}\n{fastest_row['time_minutes']:.0f} min",
    xy=(fastest_row["season"], fastest_row["time_minutes"]),
    xytext=(fastest_row["season"] - 10, fastest_row["time_minutes"] + 15),
    fontsize=8,
    color="#333333",
    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
)

# Annotate the slowest race
slowest_row = monaco_times.loc[slowest_idx]
ax.annotate(
    f"Slowest:\n{int(slowest_row['season'])}\n{slowest_row['time_minutes']:.0f} min",
    xy=(slowest_row["season"], slowest_row["time_minutes"]),
    xytext=(slowest_row["season"] + 5, slowest_row["time_minutes"] - 25),
    fontsize=8,
    color="#333333",
    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
)

ax.set_xlabel("Season (Year)")
ax.set_ylabel("Winning Time (minutes)")
ax.set_title("Monaco Grand Prix Winning Finishing Time Trend (1950–present)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)} min"))

plt.tight_layout()
save_chart("q5_monaco_finishing_time_trend.png")
plt.show()
#--------------------------------------------------------------------------------------
# ## Question 6 — Reasons Drivers Failed to Finish Races (DNF Analysis)
#
# **Business question:**
# > What are the most common reasons a driver did not finish a race (DNF),
# > ranked from most to least common?
# > Exclude records where the driver simply finished multiple laps behind.
#
# ### Approach
# 1. Remove rows where `status == "Finished"` (normal race completion)
# 2. Remove `"+X Lap(s)"` statuses using a regex pattern — these drivers
#    completed the race, just more slowly than the winner
# 3. Also remove `"Lapped"` and `"Not classified"` for the same reason
# 4. Count and rank the remaining statuses descending
#
# ### Data decisions
# - +1 Lap. are excluded per the requirement.
# -Retired is kept — it is a genuine DNF where no specific failure
#   reason was recorded by the race officials.
# - Disqualified is kept.
#   statuses in one step (future-proof, no hard-coded list of every gap).

# %%
print("=" * 60)
print("Q6 — Reasons Drivers Failed to Finish Races")
print("=" * 60)

# ── Step 1: Remove "Finished" status ──────────────────────────
dnf_data = results[results["status"] != "Finished"].copy()

laps_pattern = r"^\+\d+ Laps?$"
dnf_data = dnf_data[~dnf_data["status"].str.match(laps_pattern, na=False)]

# ── Step 3: Remove "Lapped" and "Not classified" ──────────────
# These are also drivers who completed the race (partially),
dnf_data = dnf_data[~dnf_data["status"].isin(["Lapped", "Not classified"])]

# ── Step 4: Count and rank DNF reasons ────────────────────────
dnf_counts = (
    dnf_data["status"]
    .value_counts()
    .reset_index()
    .head(20)
)
# value_counts().reset_index() gives columns: ["status", "count"]
dnf_counts.columns = ["status", "occurrences"]

print(f"\n  Total DNF records (after exclusions) : {len(dnf_data):,}")
print(f"\n  Top 20 DNF reasons:")
print(dnf_counts.to_string(index=False))

# ── Chart: Horizontal bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))

palette = sns.color_palette("flare", n_colors=len(dnf_counts))

ax.barh(
    dnf_counts["status"],
    dnf_counts["occurrences"],
    color=palette,
    edgecolor="white",
    height=0.65,
)

# Add value labels to the right of each bar (outside, for readability)
for bar in ax.patches:
    ax.text(
        bar.get_width() + 15,
        bar.get_y() + bar.get_height() / 2,
        f"{int(bar.get_width()):,}",
        va="center", ha="left",
        fontsize=8.5, color="#333333"
    )

ax.set_xlabel("Number of Occurrences")
ax.set_title(
    "Top 20 Reasons for Not Finishing a Race (DNF)\n"
    "Excluding '+Laps' records (drivers who were lapped but finished)"
)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(right=dnf_counts["occurrences"].max() * 1.18)

plt.tight_layout()
save_chart("q6_dnf_reasons.png")
plt.show()
#----------------------------------------------------------------------
# ##  Question 7 — Top 10 Races with the Highest Accident Records
#
# **Business question:**
# > Which individual races (specific race + specific year) had the most
# > accident and collision incidents across all of F1 history?
#
# ### Approach
# 1. Filter results to rows where `status` is one of the accident-related values:
#    Accident, Collision, Collision damage, Fatal accident
# 2. Group by season + race_name to identify each unique race event
# 3. Count accident-related statuses per race
# 4. Sort descending → top 10
#
# ### Data decisions
# -  include **all four accident-related statuses**, not just Accident.
# -  group by **season AND race_name** because the same circuit hosts a race
#   every year — e.g. "British Grand Prix 1975" is a different event from
#---------------------------------------------------------------------------
print("=" * 60)
print("Q7 — Top 10 Races with the Highest Accident Records")
print("=" * 60)

# ── Step 1: Define all statuses we consider "accident-related" ─
accident_statuses = ["Accident", "Collision", "Collision damage", "Fatal accident"]

# ── Step 2: Filter results to accident rows only ───────────────
accident_data = results[results["status"].isin(accident_statuses)].copy()

print(f"\n  Total accident records in dataset : {len(accident_data):,}")
print(f"\n  Status breakdown:")
print(accident_data["status"].value_counts().to_string())

# ── Step 3: Count accidents per race (season + race name) ─────
accidents_per_race = (
    accident_data
    .groupby(["season", "race_name"], as_index=False)
    .size()
    .rename(columns={"size": "accident_count"})
    .sort_values("accident_count", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# ── Step 4: Create a readable two-line label for the chart ────
accidents_per_race["race_label"] = (
    accidents_per_race["race_name"] + "\n(" + accidents_per_race["season"].astype(int).astype(str) + ")"
)

print(f"\n  Top 10 most accident-prone races:")
print(
    accidents_per_race[["season", "race_name", "accident_count"]]
    .to_string(index=False)
)

# ── Chart: Horizontal bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

bar_colours = sns.color_palette("Reds_r", n_colors=10)

ax.barh(
    accidents_per_race["race_label"],
    accidents_per_race["accident_count"],
    color=bar_colours,
    edgecolor="white",
    height=0.6,
)

# Value labels inside the bars
for bar in ax.patches:
    ax.text(
        bar.get_width() - 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{int(bar.get_width())}",
        va="center", ha="right",
        fontsize=11, color="white", fontweight="bold"
    )

ax.set_xlabel("Number of Accidents / Collisions")
ax.set_title(
    "Top 10 F1 Races with the Highest Accident Records\n"
    "(Includes: Accident, Collision, Collision Damage, Fatal Accident)"
)
ax.invert_yaxis()
ax.set_xlim(right=accidents_per_race["accident_count"].max() * 1.12)

plt.tight_layout()
save_chart("q7_top10_accident_races.png")
plt.show()

print("=" * 60)
print("✅ Analysis complete! All charts saved to charts/")
print("=" * 60)

# Final summary
summary = [
    ("Q1", "Youngest Monaco GP winner", "Lewis Hamilton — 23y 4m (2008)"),
    ("Q2", "Top constructor by wins",   "Ferrari — 223 all-time wins"),
    ("Q3", "Top Q2 country",            "Germany — 1,006 Q2 appearances"),
    ("Q4", "Longest career",            "Fernando Alonso — 26 seasons"),
    ("Q5", "Fastest Monaco race",       "1984 — 61 min winning time"),
    ("Q6", "#1 DNF reason",             "Engine failure — 2,011 retirements"),
    ("Q7", "Most accident-prone race",  "1975 British GP — 16 incidents"),
]

print()
for q, label, finding in summary:
    print(f"  [{q}] {label:<35} → {finding}")
print()