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
| **Population Share in Broad Region** | `POPESTIMATE` from ACS/PEP sources | `Share = (Mean POPESTIMATE for Focus / Mean POPESTIMATE for Broad) * 100` | Calculates the Focus geography's percentage share of the Broad geography's population. |
| **Change in Population Share** | YoY Share Comparison | `Change = Share (latest) - Share (previous)` | Computes the percentage point change in population share year-over-year. |
| **Cumulative Population Change** | Metric name includes "cumulative" and "population" | Absolute change:<br>`Change = POP (latest) - POP (1990)`<br>Percentage change:<br>`Change = [ (POP (latest) - POP (1990)) / POP (1990) ] * 100` | Measures growth or decline in population since the 1990 baseline. |
| **Primary Driver of Change** | `components_of_change` | Finds the component with the largest absolute value in the latest year among:<br>• `NATURALCHG` (Natural Change)<br>• `DOMESTICMIG` (Domestic Migration)<br>• `INTERNATIONALMIG` (International Migration) | Pinpoints the primary force behind a geography's population shifts. |
| **Largest Group (Population Pyramid)** | `population_pyramid` | Selects category `C` that maximizes:<br>`Sum of Population Count for C` in the latest year. | Identifies the most populous age or race-ethnicity group. |
| **Largest Change in Group (Pyramid)** | `population_pyramid` YoY change | Selects category `C` that maximizes:<br>`Count (C, latest) - Count (C, previous)` | Identifies the cohort experiencing the largest absolute growth or decline. |
| **Largest Group (ACS Series)** | Multi-label cohort proportion | Identifies the Label `L` with the maximum proportion value in the latest year. | Identifies the largest housing type or demographic cohort from ACS. |
| **Largest Change in Group (ACS)** | Multi-label cohort YoY change | Identifies the Label `L` that maximizes:<br>`Value (L, latest) - Value (L, previous)` | Identifies the cohort with the largest year-over-year percentage point shift. |
| **Standard YoY Change** | Single-field metric | `Change = Value (latest) - Value (previous)` | Standard simple year-over-year numeric or percentage point change. |

---

### 2. Housing Affordability Index (HAI) & Mortgage Indicator Formulas

For metrics sourced from the `HAI-CPI Table`, the pipeline computes custom affordability indicators based on the National Association of Realtors (NAR) methodology:

#### Monthly Mortgage Payment Formula
The estimated monthly mortgage payment assumes a **20% down payment** (financing 80% of the median listing price) on a standard **30-year fixed mortgage**:
*   `Principal (P) = Median Listing Price * 0.80`
*   `Monthly Interest Rate (r) = (Monthly Avg 30yr Rate / 100) / 12`
*   `Number of Months (n) = 360`
*   `Monthly Payment = P * [ r * (1 + r)^n ] / [ (1 + r)^n - 1 ]`

#### Qualifying Annual Income
Assumes a standard conservative qualifying debt-to-income (DTI) ratio where mortgage payments are capped at **20% of gross income**:
*   `Qualifying Income = (Monthly Payment * 12) / 0.2`

#### Housing Affordability Index (HAI)
The HAI measures whether a family earning the median household income has enough income to qualify for a mortgage on a median-priced home:
*   `HAI = (Median Household Income / Qualifying Income) * 100`
*   **HAI = 100**: A family earning the median income has exactly the income required to qualify for a median-priced home.
*   **HAI > 100**: Homeownership is increasingly affordable (family income exceeds qualifying income).
*   **HAI < 100**: Homeownership is increasingly unaffordable.

---

## How It Works: The Pipeline Architecture Map

To ensure that the summaries are both beautifully written and 100% mathematically accurate, the pipeline operates in four distinct stages. Here is a simple map of how the data flows from start to finish:

### 1. Data Ingestion (The Foundation)
*   **What it does:** The script reads the "Blueprint" (your Excel sheet) to know exactly which metrics to generate and which ones to skip. 
*   **The Rules:** 
    *   It extracts the **exact geographical names** directly from your input data (e.g., "Fife, WA", "Seattle MSA"). This creates a master list of valid names to prevent the AI from making up or misspelling places later.
    *   It standardizes all incoming datasets (ACS, Population Pyramid, Housing, etc.) so they can be compared easily.

### 2. The Deterministic Cyborg Engine (The Hard Math)
*   **What it does:** Before the AI ever sees the data, the code does all the heavy lifting using hardcoded, traditional math. This prevents "AI hallucinations" (where an AI confidently makes up fake numbers).
*   **The Rules (Formatting & Math):**
    *   It calculates absolute values, year-over-year changes, and comparative differences between the focus area, its peers, the state, and the nation.
    *   **Dollar Rounding:** If a dollar value is under $10, it keeps two decimal places (e.g., `$9.32`). If it is $10 or more, it rounds to the nearest whole dollar (e.g., `$15`).
    *   **Percentages & Demographics:** Rounded neatly to one decimal place (e.g., `31.3%` or `+0.7 percentage points`).
    *   **Large Numbers:** Values over 1,000,000 are converted to readable formats (e.g., `1.50 million`).

### 3. The AI Synthesizer (The Writer)
*   **What it does:** The bulletproof math from Step 2 is handed over to the AI (Gemini or Ollama). The AI acts as an executive data analyst, weaving those raw numbers into a fluid, professional paragraph.
*   **What goes into the AI:** 
    *   A strict prompt with a 150-word limit.
    *   The "Math Context" containing the exact, pre-calculated numbers (so the AI doesn't have to do any math itself).
    *   The authoritative list of valid geography names.
*   **The Rules (Copyediting):**
    *   Geography names must use Title Case and match the input list exactly (e.g., "Seattle MSA", never "SEATTLE MSA").
    *   Demographic categories and drivers of change (like "natural change", "female", "white alone") must be lowercase, unless they start a sentence.
    *   Internal database column names (like `calc-multi-family`) must be translated into natural phrasing.
    *   No conversational filler ("Interestingly, the data shows..."). It must be declarative and professional.

### 4. The Quality Control Layer (The Safety Net)
*   **What it does:** Even with strict instructions, AI can sometimes make grammatical typos. This final, hardcoded layer scans the AI's finished text right before exporting to forcefully correct any slip-ups.
*   **The Rules (Text Enforcement):**
    *   **Orphaned City Fix:** If the AI writes a standalone city name (like "Fife") but the valid input list says it should be "Fife, WA", the script automatically appends the state.
    *   **State Trailing Comma Fix:** If a state abbreviation (like WA or TX) is followed by another word in the middle of a sentence, the script automatically injects a grammatically correct trailing comma (turning *"Fife, WA experienced growth"* into *"Fife, WA, experienced growth"*).

---

## Repository Structure

```
├── README.md                  # This documentation file
├── ai_summary_pipeline.py     # Main Python pipeline module & CLI entry point
├── Metric Topics (DRAFT).xlsx # Excel blueprint containing metric configurations (sheet AI Summary)
├── examples/                  # Example datasets (ACS, Components of Change, Pyramid, HAI, CPI, Mortgage)
├── archive/                   # Archived notebooks folder
└── .gitignore                 # Git ignore configurations
```
