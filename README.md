# 🏎️ Formula 1 Racing — Exploratory Data Analysis

A clean, beginner-friendly data analysis project built with Python.
Answers 7 business questions about Formula 1 racing using real data from 1950 to 2026.

---

## 📂 Project Structure

```
f1_analysis/
├── data/                          ← Raw CSV files (place them here)
│   ├── f1_results.csv
│   ├── f1_drivers.csv
│   ├── f1_races.csv
│   ├── f1_qualifying.csv
│   ├── f1_circuits.csv
│   ├── f1_constructors.csv
│   └── f1_constructor_standings.csv
├── charts/                        ← Auto-generated chart images
│   ├── q1_youngest_monaco_winner.png
│   ├── q2_top10_constructor_wins.png
│   ├── q3_q2_concentration_by_country.png
│   ├── q4_longest_careers.png
│   ├── q5_monaco_finishing_time_trend.png
│   ├── q6_dnf_reasons.png
│   └── q7_top10_accident_races.png
├── f1_analysis_vscode.py          ← Main analysis file (run this)
└── README.md                      ← This file
```

---

## 🚀 How to Run in VS Code

### Prerequisites

Install the required Python libraries (if not already installed):

```bash
pip install pandas matplotlib seaborn
```

Install the following VS Code extensions:
- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

### Running the analysis

1. Open `f1_analysis_vscode.py` in VS Code
2. VS Code will automatically show **"Run Cell"** buttons above every `# %%` marker
3. Click **"Run All Cells"** in the toolbar to run the full analysis
4. Charts appear inline in the **Interactive Panel** on the right side
5. All charts are also saved to the `charts/` folder automatically

> **Tip:** If you don't see "Run Cell" buttons, make sure the Jupyter extension
> is installed and that your Python environment is selected (bottom-left corner of VS Code).

---

## ❓ Business Questions Answered

| # | Question | Key Finding |
|---|----------|------------|
| 1 | Youngest Monaco GP winner ever | **Lewis Hamilton** — 23y 4m (2008) |
| 2 | Top 10 constructors by race wins | **Ferrari** leads with 223 all-time wins |
| 3 | Q2 qualification by country (top 10) | **Germany** leads with 1,006 Q2 appearances |
| 4 | Drivers with longest F1 careers | **Fernando Alonso** — 26 seasons (2001–2026) |
| 5 | Monaco GP finishing time trend | From **193 min** (1950) down to **61 min** (1984) |
| 6 | Top DNF (did not finish) reasons | **Engine failure** — 2,011 retirements |
| 7 | Races with most accident records | **1975 British GP** — 16 incidents in one race |

---

## 📊 Dataset Overview

| File | Rows | Description |
|------|------|-------------|
| `f1_results.csv` | 25,939 | Every driver's result for every race |
| `f1_drivers.csv` | 879 | Driver personal info (DOB, nationality) |
| `f1_races.csv` | 1,171 | Race calendar (date, circuit, season) |
| `f1_qualifying.csv` | 11,036 | Q1/Q2/Q3 lap times per driver per race |
| `f1_circuits.csv` | 78 | Circuit details (name, country) |
| `f1_constructors.csv` | 214 | Constructor (team) info |
| `f1_constructor_standings.csv` | 924 | Season-end constructor points and wins |

**Season coverage:** 1950–2026  
**Total race result records:** 25,939

---

## 🛠️ Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 1.5 | Data loading, cleaning, transformation |
| `matplotlib` | ≥ 3.5 | Base charting engine |
| `seaborn` | ≥ 0.12 | Styled, higher-level charts |

No machine learning, no complex SQL, no advanced Python patterns.
The code is intentionally written to be **readable by beginners**.

---

## 🧹 Data Cleaning Decisions

| Decision | Reason |
|----------|--------|
| `time` missing in 67% of results rows | Expected — only the race winner has an absolute time; others show a gap like `+1.5s` |
| Drop rows with missing `dob` for Q1 | Can't calculate driver age without a birthdate |
| Q2/Q3 qualifying times mostly null pre-2006 | Modern qualifying format only introduced in 2006 |
| Exclude `+X Lap` statuses from DNF analysis | Lapped drivers did finish the race — they just completed fewer laps |
| Group by `season + race_name` for Q7 | Same race name runs every year; season is needed to identify each unique event |

---

## 💡 Key Insights

- **F1 cars are 3× faster at Monaco** than in 1950 — from 193-minute races to 61-minute races
- **Ferrari's 223 wins** is a record built over 7 decades; Mercedes achieved 125 wins in just ~15 years
- **Engine failure** caused more retirements than any other reason — 2,011 times across F1 history
- **Fernando Alonso's 26-season career** spans nearly half of F1's entire history as a sport
- **Germany and Britain** produce the most Q2-qualifying drivers by a wide margin

---

*Dataset sourced from public F1 historical records. Analysis is for educational and portfolio purposes.*