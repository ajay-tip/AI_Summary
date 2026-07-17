# AI Summary Generation Pipeline

The **AI Summary Generation Pipeline** is a modular Python-based data processing and text synthesis tool. It standardizes demographic and economic datasets, computes deterministic comparative metrics against peers, benchmarks, and national averages, and leverages Large Language Models (Gemini / Ollama) to synthesize professional, qualitative summary insights for dashboards.

---

## Features

- **Dual Model Support**: Supports both **Gemini API** (`gemini-2.5-flash` default) and **Ollama** (`gemma3`, etc.) for generating natural language insights.
- **Deterministic Cyborg Template Engine**: Automatically calculates internal, broad geographic, benchmark, and peer-detailed insights using math-based template rules before synthesizing them using LLMs.
- **Flag-based Filtering**: Reads the metric blueprint from Excel (sheet `AI Summary` by default) and automatically filters out rows where `Flag for Use` is set to `False`.
- **Concise LLM Output**: Limits generated summary text to 150 words for overall insight, topic summary, and complete executive summary.
- **Population Source Preference**: Prefers PEP-sourced population values when available and computes focus-to-broad population share and share change metrics where applicable.
- **Dynamic File Configurations**: Accepts custom inputs for ACS, Components of Change, Population Pyramid data, and the blueprint sheets.
- **Importable & CLI-Ready**: Designed both as an importable module and a standalone CLI tool.

---

## Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install polars pandas openpyxl
```

### LLM Setup

#### 1. Gemini Mode (Default)
Install the official Google GenAI SDK:
```bash
pip install google-genai
```
Configure your environment variables with your API key:
- **Windows (PowerShell)**:
  ```powershell
  $env:GEMINI_API_KEY="your-api-key-here"
  ```
- **Linux/macOS**:
  ```bash
  export GEMINI_API_KEY="your-api-key-here"
  ```

#### 2. Ollama Mode (Alternative)
Install the Ollama client package:
```bash
pip install ollama
```
Make sure you have Ollama running locally and have pulled your target model (e.g. `gemma3`):
```bash
ollama run gemma3
```

---

## Usage

### 1. As an Imported Function
You can clone this repository and import `run_pipeline` directly into your codebase:

```python
from ai_summary_pipeline import run_pipeline

# Execute using default Gemini API configurations
df_results = run_pipeline(
    blueprint_path="Metric Topics (DRAFT).xlsx",
    acs_path="ACS_Series_Polars.csv",
    components_path="components_of_change (4).csv",
    pyramid_path="population_pyramid.csv",
        sheet_name="AI Summary",
print(df_results.head())
```

### 2. Via Command Line (CLI)
Run the script directly from your terminal:

```bash
# Using default Gemini configuration:
python ai_summary_pipeline.py --sheet "AI Summary" --output final_dashboard_data.csv

# Using Ollama locally with gemma3 model:
python ai_summary_pipeline.py --mode ollama --model-name gemma3 --sheet "AI Summary"
```

To see all available CLI options:
```bash
python ai_summary_pipeline.py --help
```

---

## Inputs Schema & Data Requirements

The pipeline expects the following input files:
1. **Blueprint Excel** (`Metric Topics (DRAFT).xlsx`): Contains target metrics, descriptions, metric types, comparison periods, variables mapping, and a `Flag for Use` column to enable or disable processing.
2. **ACS Data** (`ACS_Series_Polars.csv`): Geographic ACS timeseries data.
3. **Components of Change** (`components_of_change (4).csv`): Geographic components of population change.
4. **Population Pyramid** (`population_pyramid.csv`): Breakdown of age and race groups across years.

---

## Metric Calculations and Formulas

The pipeline processes diverse demographic and economic indicators. Below is the reference table of exact mathematical formulas and logical workflows utilized by the Cyborg Engine:

### 1. General & Demographic Metrics

| Metric Category | Source / Condition | Mathematical Formula / Logic | Description |
| :--- | :--- | :--- | :--- |
| **Population Share in Broad Region** | `POPESTIMATE` from ACS/PEP sources | $$\text{Share} = \frac{\text{Mean POPESTIMATE for Focus}}{\text{Mean POPESTIMATE for Broad}} \times 100$$ | Calculates the Focus geography's percentage share of the Broad geography's population. |
| **Change in Population Share** | YoY Share Comparison | $$\text{Change} = \text{Share}_{\text{latest}} - \text{Share}_{\text{prev}}$$ | Computes the percentage point change in population share year-over-year. |
| **Cumulative Population Change** | Metric name includes "cumulative" and "population" | Absolute change: $$\text{Change} = \text{POP}_{\text{latest}} - \text{POP}_{1990}$$ <br> Percentage change: $$\text{Change} = \frac{\text{POP}_{\text{latest}} - \text{POP}_{1990}}{\text{POP}_{1990}} \times 100$$ | Measures growth or decline in population since the 1990 baseline. |
| **Primary Driver of Change** | `components_of_change` | Finds the component with the largest absolute value in the latest year among: <br> • `NATURALCHG` (Natural Change) <br> • `DOMESTICMIG` (Domestic Migration) <br> • `INTERNATIONALMIG` (International Migration) | Pinpoints the primary force behind a geography's population shifts. |
| **Largest Group (Population Pyramid)** | `population_pyramid` | Selects category $C$ that maximizes: $$\sum \text{Population Count}_C$$ in the latest year. | Identifies the most populous age or race-ethnicity group. |
| **Largest Change in Group (Pyramid)** | `population_pyramid` YoY change | Selects category $C$ that maximizes: $$\text{Count}_{C, \text{latest}} - \text{Count}_{C, \text{prev}}$$ | Identifies the cohort experiencing the largest absolute growth or decline. |
| **Largest Group (ACS Series)** | Multi-label cohort proportion | Identifies the Label $L$ with the maximum proportion value in the latest year. | Identifies the largest housing type or demographic cohort from ACS. |
| **Largest Change in Group (ACS)** | Multi-label cohort YoY change | Identifies the Label $L$ that maximizes: $$\text{Value}_{L, \text{latest}} - \text{Value}_{L, \text{prev}}$$ | Identifies the cohort with the largest year-over-year percentage point shift. |
| **Standard YoY Change** | Single-field metric | $$\text{Change} = \text{Value}_{\text{latest}} - \text{Value}_{\text{prev}}$$ | Standard simple year-over-year numeric or percentage point change. |

---

### 2. Housing Affordability Index (HAI) & Mortgage Indicator Formulas

For metrics sourced from the `HAI-CPI Table`, the pipeline computes custom affordability indicators based on the National Association of Realtors (NAR) methodology:

#### Monthly Mortgage Payment Formula
The estimated monthly mortgage payment assumes a **20% down payment** (financing 80% of the median listing price) on a standard **30-year fixed mortgage**:
$$\text{Principal (P)} = \text{Median Listing Price} \times 0.80$$
$$\text{Monthly Interest Rate (r)} = \frac{\text{Monthly Avg 30yr Rate}}{100 \times 12}$$
$$\text{Number of Months (n)} = 360$$
$$\text{Monthly Payment} = P \times \frac{r(1+r)^n}{(1+r)^n - 1}$$

#### Qualifying Annual Income
Assumes a standard conservative qualifying debt-to-income (DTI) ratio where mortgage payments are capped at **25% of gross income**:
$$\text{Qualifying Income} = \text{Monthly Payment} \times 12 \times 4$$

#### Housing Affordability Index (HAI)
The HAI measures whether a family earning the median household income has enough income to qualify for a mortgage on a median-priced home:
$$\text{HAI} = \frac{\text{Median Household Income}}{\text{Qualifying Income}} \times 100$$
*   **HAI = 100**: A family earning the median income has exactly the income required to qualify for a median-priced home.
*   **HAI > 100**: Homeownership is increasingly affordable (family income exceeds qualifying income).
*   **HAI < 100**: Homeownership is increasingly unaffordable.

---

## Repository Structure

```
├── README.md                  # This documentation file
├── ai_summary_pipeline.py     # Main Python pipeline module & CLI entry point
├── Metric Topics (DRAFT).xlsx # Excel blueprint containing metric configurations (sheet AI Summary)
├── ACS_Series_Polars.csv      # ACS timeseries source dataset
├── components_of_change (4).csv # Population change components dataset
├── population_pyramid.csv     # Population age/race breakdown dataset
├── archive/                   # Archived notebooks folder
│   ├── AI Summary Testing.ipynb
│   └── AI Summary Testing - Deterministic + LLM.ipynb
└── .gitignore                 # Git ignore configurations
```
