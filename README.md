# AI Summary Generation Pipeline

The **AI Summary Generation Pipeline** is a modular Python-based data processing and text synthesis tool. It standardizes demographic and economic datasets, computes deterministic comparative metrics against peers, benchmarks, and national averages, and leverages Large Language Models (Gemini / Ollama) to synthesize professional, qualitative summary insights for dashboards.

---

## Features

- **Dual Model Support**: Supports both **Gemini API** (`gemini-2.5-flash` default) and **Ollama** (`gemma3`, etc.) for generating natural language insights.
- **Deterministic Cyborg Template Engine**: Automatically calculates internal, broad geographic, benchmark, and peer-detailed insights using math-based template rules before synthesizing them using LLMs.
- **Flag-based Filtering**: Reads the metric blueprint from Excel (sheet `v3` by default) and automatically filters out rows where `Flag for Use` is set to `False`.
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
    sheet_name="v3",
    mode="gemini",
    gemini_model="gemini-2.5-flash",
    output_path="dashboard_data_debug_v4.csv"
)

print(df_results.head())
```

### 2. Via Command Line (CLI)
Run the script directly from your terminal:

```bash
# Using default Gemini configuration:
python ai_summary_pipeline.py --sheet v3 --output final_dashboard_data.csv

# Using Ollama locally with gemma3 model:
python ai_summary_pipeline.py --mode ollama --model-name gemma3 --sheet v3
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

## Repository Structure

```
├── README.md                  # This documentation file
├── ai_summary_pipeline.py     # Main Python pipeline module & CLI entry point
├── Metric Topics (DRAFT).xlsx # Excel blueprint containing metric configurations (sheet v3)
├── ACS_Series_Polars.csv      # ACS timeseries source dataset
├── components_of_change (4).csv # Population change components dataset
├── population_pyramid.csv     # Population age/race breakdown dataset
├── archive/                   # Archived notebooks folder
│   ├── AI Summary Testing.ipynb
│   └── AI Summary Testing - Deterministic + LLM.ipynb
└── .gitignore                 # Git ignore configurations
```
