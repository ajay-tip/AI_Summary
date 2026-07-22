import polars as pl
import pandas as pd
import time
import json
import random 
import re 
import os
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Safe/dynamic imports for LLM SDKs to allow running the pipeline
# without requiring all libraries to be installed if they aren't used.
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class AISummaryPipeline:
    def __init__(self, mode: str = "gemini", model_name: str = "gemma3", 
                 gemini_model: str = "gemini-2.5-flash", api_key: str = None,
                 vertexai: bool = None, project: str = None, location: str = None):
        """
        Initialize the AI Summary Pipeline.
        
        Args:
            mode (str): Model provider, either "gemini" or "ollama". Default is "gemini".
            model_name (str): Model name for Ollama mode (default: "gemma3").
            gemini_model (str): Model name for Gemini mode (default: "gemini-2.5-flash").
            api_key (str, optional): API Key for Gemini Developer API. If None, loaded from GEMINI_API_KEY.
            vertexai (bool, optional): Whether to use Vertex AI backend instead of Developer API. 
                                      If None, auto-detected from GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT.
            project (str, optional): Google Cloud project ID (for Vertex AI mode).
            location (str, optional): Google Cloud region (for Vertex AI mode, e.g. "us-central1").
        """
        self.mode = mode.lower().strip()
        self.model_name = model_name
        self.gemini_model = gemini_model
        self.api_key = api_key
        
        # Auto-detect Vertex AI mode if credentials/project are set in the environment
        if vertexai is None:
            has_credentials = "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
            has_project = "GOOGLE_CLOUD_PROJECT" in os.environ
            vertexai = has_credentials or has_project
            
        self.vertexai = vertexai
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        self.gemini_client = None
        
        # Period tracking for dynamic placeholders (anchored to primary data source)
        self.global_latest_month = None
        self.global_prev_month = None

        if self.mode == "gemini":
            if not HAS_GENAI:
                raise ImportError(
                    "The 'google-genai' package is required for 'gemini' mode but is not installed. "
                    "Please install it using: pip install google-genai"
                )
            
            client_kwargs = {}
            if self.vertexai:
                client_kwargs["vertexai"] = True
                if self.project:
                    client_kwargs["project"] = self.project
                if self.location:
                    client_kwargs["location"] = self.location
            else:
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                    
            self.gemini_client = genai.Client(**client_kwargs)
            
        elif self.mode == "ollama":
            if not HAS_OLLAMA:
                raise ImportError(
                    "The 'ollama' package is required for 'ollama' mode but is not installed. "
                    "Please install it using: pip install ollama"
                )
    # ==========================================
    # 1. STANDARDIZATION & FORMATTING
    # ==========================================
    def standardize_dataset(self, df: pl.DataFrame, source_type: str) -> pl.DataFrame:
        # Aggressively strip column names to prevent missed lookups
        df.columns = [c.strip() for c in df.columns]
        
        # Adaptable geography standardization: Rely entirely on 'Display Name' if available.
        # NO HARDCODED NAMES.
        if "Display Name" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("Display Name").is_not_null() & (pl.col("Display Name").cast(pl.Utf8).str.strip_chars() != ""))
                .then(pl.col("Display Name").cast(pl.Utf8).str.strip_chars())
                .otherwise(pl.col("NAME").cast(pl.Utf8).str.strip_chars())
                .alias("NAME")
            )
        else:
            # Fallback to just cleaning the existing NAME column
            if "NAME" in df.columns:
                df = df.with_columns(pl.col("NAME").cast(pl.Utf8).str.strip_chars())
            elif "Name" in df.columns:
                df = df.with_columns(pl.col("Name").cast(pl.Utf8).str.strip_chars().alias("NAME"))

        if source_type == "ACS":
            return df.select([pl.col("Year").cast(pl.Int64), pl.col("Cleaned Field Name").alias("Metric"), pl.col("NAME"), pl.col("Value").cast(pl.Float64), pl.col("Role")])
        elif source_type == "COMPONENTS":
            return df.select([
                pl.col("Year").cast(pl.Int64),
                pl.col("Value Type").alias("Metric"),
                pl.col("NAME"),
                pl.col("Values").alias("Value").cast(pl.Float64),
                pl.col("Role"),
                pl.col("Source"),
                pl.col("Dataset"),
                pl.col("Geography Level")
            ])
        elif source_type == "POP_PYRAMID":
            return df.select([pl.col("YEAR").cast(pl.Int64).alias("Year"), (pl.col("Variable Group Description") + " (" + pl.col("Age Group Description") + ")").alias("Metric"), pl.col("NAME"), pl.col("Values").alias("Value").cast(pl.Float64), pl.col("Role")])
        elif source_type == "HAI":
            field_map = {
                "Calc-Median HH Income": "Calc-Median HH Income",
                "Calc-Median Value of Owned Units": "Calc-Median Value of Owned Units",
                "Calc-Median Monthly Rent": "Calc-Median Monthly Rent",
                "median_listing_price": "Median Listing Price"
            }
            # Standardize column names to lowercase for robust mapping
            col_map = {c.lower(): c for c in df.columns}
            
            # Generate MonthKey from various possible sources
            if "month_date_yyyymm" in col_map:
                df = df.with_columns(pl.col(col_map["month_date_yyyymm"]).cast(pl.Int64).alias("MonthKey"))
            elif "year" in col_map and "month" in col_map:
                df = df.with_columns((pl.col(col_map["year"]).cast(pl.Int64) * 100 + pl.col(col_map["month"]).cast(pl.Int64)).alias("MonthKey"))
            else:
                # Fallback: Use Year alone if month is missing
                df = df.with_columns((pl.col(col_map.get("year", "ACS_Year")).cast(pl.Int64) * 100 + 1).alias("MonthKey"))

            df = df.with_columns(
                pl.col(col_map.get("year", "ACS_Year")).cast(pl.Int64).alias("Year"),
                pl.col("NAME").cast(pl.Utf8).str.strip_chars(),
                pl.col("Role").cast(pl.Utf8).str.strip_chars()
            )
            
            metric_frames = []
            for raw_field, metric_name in field_map.items():
                # Case-insensitive check
                actual_col = next((c for c in df.columns if c.lower() == raw_field.lower()), None)
                if actual_col:
                    metric_frames.append(
                        df.select(["Year", "NAME", "Role", "MonthKey", pl.col(actual_col).alias("Value")])
                          .with_columns(
                              pl.lit(metric_name).alias("Metric"),
                              pl.lit("HAI").alias("Source"),
                              pl.lit("HAI-CPI Table").alias("Dataset"),
                              pl.lit("HAI").alias("Geography Level")
                          )
                          .filter(pl.col("Value").is_not_null())
                          .with_columns(pl.col("Value").cast(pl.Float64))
                          .sort("MonthKey", descending=True)
                    )
            return pl.concat(metric_frames) if metric_frames else pl.DataFrame()
        elif source_type == "CPI":
            frames = []
            frames.append(df.select([
                pl.col("Year").cast(pl.Int64),
                pl.lit("Average CPI (Annual)").alias("Metric"),
                pl.lit("United States").alias("NAME"),
                pl.col("Average CPI (Annual)").cast(pl.Float64).alias("Value"),
                pl.lit("Benchmark").alias("Role"),
                pl.lit("CPI").alias("Source"),
                pl.lit("CPI Table").alias("Dataset"),
                pl.lit("National").alias("Geography Level"),
                pl.lit(None).cast(pl.Int64).alias("MonthKey")
            ]))
            if "Month" in df.columns and "CPI" in df.columns:
                frames.append(df.select([
                    pl.col("Year").cast(pl.Int64),
                    pl.lit("CPI (Monthly)").alias("Metric"),
                    pl.lit("United States").alias("NAME"),
                    pl.col("CPI").cast(pl.Float64).alias("Value"),
                    pl.lit("Benchmark").alias("Role"),
                    pl.lit("CPI").alias("Source"),
                    pl.lit("CPI Table").alias("Dataset"),
                    pl.lit("National").alias("Geography Level"),
                    (pl.col("Year").cast(pl.Int64) * 100 + pl.col("Month").cast(pl.Int64)).alias("MonthKey")
                ]))
            return pl.concat(frames, how="diagonal_relaxed")
        elif source_type == "MORTGAGE":
            if "Monthly Avg 30yr" not in df.columns:
                return pl.DataFrame()
            return df.select([
                pl.col("Year").cast(pl.Int64),
                pl.lit("Monthly Avg 30yr").alias("Metric"),
                pl.lit("United States").alias("NAME"),
                pl.col("Monthly Avg 30yr").cast(pl.Float64).alias("Value"),
                pl.lit("Benchmark").alias("Role"),
                pl.lit("Mortgage Rates").alias("Source"),
                pl.lit("Mortgage Table").alias("Dataset"),
                pl.lit("National").alias("Geography Level"),
                (pl.col("Year").cast(pl.Int64) * 100 + pl.col("Month").cast(pl.Int64)).alias("MonthKey")
            ])

    def format_value(self, value, metric_name, m_type="", is_percentage_override=None):
        try:
            val = float(value)
            if val != val: return "N/A"
        except: 
            return str(value)
        
        m_lower = str(metric_name).lower()
        type_lower = str(m_type).lower()
        
        is_percentage = (
            "percent" in type_lower or 
            "rate" in type_lower or 
            "categorical" in type_lower or
            "largest" in m_lower or
            ("percent" in m_lower and "numeric" not in type_lower)
        )
        if is_percentage_override is not None:
            is_percentage = is_percentage_override
            
        if is_percentage and abs(val) > 1000:
            is_percentage = False
        
        if is_percentage:
            if "change" in m_lower and val > 0:
                return f"+{val:.1f}%"
            return f"{val:.1f}%"
                
        elif "age" in m_lower and "largest" not in m_lower:  
            if val == 0: return "0.0"
            return f"+{val:.1f}" if ("change" in m_lower and val > 0) else f"{val:.1f}"
        elif "income" in m_lower or "cost" in m_lower: 
            return f"${val:,.0f}"
        elif "change" in m_lower or "net" in m_lower or "mig" in m_lower or "driver" in m_lower: 
            if val == 0: return "0"
            return f"+{val:,.0f}" if val > 0 else f"{val:,.0f}"
        else: 
            if val >= 1_000_000: return f"{val/1_000_000:.2f} million"
            return f"{val:,.0f}"

    def format_period_placeholders(self, text):
        """
        Replaces bracketed month placeholders with actual formatted dates.
        Example: [Current Month] -> April 2026
                 [Previous-12 Month] -> April 2025
        """
        if not isinstance(text, str) or not text:
            return text

        # Use global keys set from the HAI data if available, otherwise fallback to system time
        if self.global_latest_month:
            year = self.global_latest_month // 100
            month = self.global_latest_month % 100
            current_date = datetime(year, month, 1)
        else:
            current_date = datetime.now()
        
        if self.global_prev_month:
            year = self.global_prev_month // 100
            month = self.global_prev_month % 100
            previous_12_date = datetime(year, month, 1)
        else:
            # Fallback: exactly 12 months before current
            previous_12_date = current_date - relativedelta(months=12)

        # Format dates as "Month Year" (e.g., "June 2025")
        current_month_str = current_date.strftime('%B %Y')
        prev_12_month_str = previous_12_date.strftime('%B %Y')

        # Dictionary of placeholders to replace
        replacements = {
            '[Current Month]': current_month_str,
            '[Previous-12 Month]': prev_12_month_str
        }

        # Apply replacements
        for placeholder, actual_date in replacements.items():
            text = text.replace(placeholder, actual_date)

        return text

    # ==========================================
    # 2. DETERMINISTIC TEMPLATE ENGINE (Cyborg)
    # ==========================================
    def is_missing_value(self, value):
        if value is None:
            return True
        if isinstance(value, str) and str(value).strip().upper() in ["", "N/A", "NA"]:
            return True
        try:
            if isinstance(value, float) and math.isnan(value):
                return True
        except Exception:
            pass
        return False

    def generate_internal_insight(self, fn, f_val, metric_name, m_type, is_yoy, cy, py, is_cat, f_cat):
        if self.is_missing_value(f_val):
            return "N/A"
        f_str = self.format_value(f_val, metric_name, m_type)
        m_clean = metric_name.lower().replace("percentage ", "").replace("percent ", "")
        
        if is_cat:
            return f"In {cy}, {fn}'s primary {m_clean} was {f_cat} ({f_str})."
        if is_yoy:
            return f"Between {py} and {cy}, {fn} experienced a {m_clean} of {f_str}."
        else:
            return f"In {cy}, {fn} recorded a {m_clean} of {f_str}."

    def generate_comparative_insight(self, fn, comparisons, f_val, metric_name, m_type, is_cat, f_cat):
        if f_val is None or not comparisons: return "N/A"
        
        f_str = self.format_value(f_val, metric_name, m_type)
        m_clean = metric_name.lower().replace("percentage ", "").replace("percent ", "")
        
        parts = []
        for comp in comparisons:
            c_val = comp.get("Raw_Value")
            cn = comp.get("Name", "")
            c_cat = comp.get("Category")
            if c_val is None: continue
            
            c_str = self.format_value(c_val, metric_name, m_type)
            cn_clean = cn.replace("Peers (Average)", "its peer average").replace("Peers (Combined)", "its combined peers")

            if is_cat:
                if f_cat == c_cat:
                    parts.append(f"{fn} and {cn_clean} both share {f_cat} as their primary {m_clean}, with {fn} at {f_str} and {cn_clean} at {c_str}")
                else:
                    parts.append(f"while {fn}'s primary {m_clean} is {f_cat} ({f_str}), {cn_clean} is primarily driven by {c_cat} ({c_str})")
            else:
                try:
                    f = float(f_val)
                    c = float(c_val)
                    is_percentage = "%" in f_str or "%" in c_str or "percent" in str(m_type).lower() or "rate" in str(m_type).lower() or "percent" in m_clean
                    if is_percentage and (abs(f) > 1000 or abs(c) > 1000): is_percentage = False

                    if f == c:
                        parts.append(f"is identical to {cn_clean}")
                        continue
                        
                    diff = f - c
                    if is_percentage:
                        direction = "higher" if diff > 0 else "lower"
                        parts.append(f"is {abs(diff):.1f} percentage points {direction} than {cn_clean} ({c_str})")
                    elif "change" in m_clean or "net" in m_clean or "mig" in m_clean:
                        direction = "larger" if diff > 0 else "smaller"
                        parts.append(f"represents a {direction} change compared to {cn_clean} ({c_str})")
                    else:
                        direction = "higher" if "age" in m_clean else ("larger" if f > c else "smaller")
                        parts.append(f"is {direction} than {cn_clean} ({c_str})")
                except:
                    parts.append(f"is {f_str}, compared to {c_str} in {cn_clean}")
                    
        if not parts: return "N/A"
        
        if is_cat:
            draft = parts[0]
            if len(parts) > 1:
                for p in parts[1:]:
                    draft += f". {p[:1].upper()}{p[1:]}"
            return draft + "."
        else:
            draft = f"{fn}'s {m_clean} of {f_str} {parts[0]}"
            if len(parts) > 1:
                if len(parts) == 2:
                    draft += f" and {parts[1]}"
                else:
                    for p in parts[1:-1]:
                        draft += f", {p}"
                    draft += f", and {parts[-1]}"
            return draft + "."

    def generate_categorical_peer_detailed_insight(self, f_val, f_cat, peer_details, peer_comb_cat, fn, metric_name, m_type, is_percentage_override=None):
        """Deterministically generates detailed category comparisons to individual peers."""
        try:
            f_str = self.format_value(f_val, metric_name, m_type, is_percentage_override=is_percentage_override)
            same_cat, diff_cat = [], []
            debug_math = f"Focus ({fn}) Cat: {f_cat} ({f_str}) | "
            
            for p in peer_details:
                if fn.lower() == p["Name"].lower() or p["Name"].lower() in fn.lower(): continue
                p_val = float(p["Value"])
                p_str = self.format_value(p_val, metric_name, m_type, is_percentage_override=is_percentage_override)
                peer_obj = {"name": p["Name"], "cat": p["Category"], "val": p_val, "str": p_str}
                
                debug_math += f"[{p['Name']}: {p['Category']} ({p_str})] "
                if p["Category"] == f_cat: same_cat.append(peer_obj)
                else: diff_cat.append(peer_obj)
                    
            # Sort by actual numerical size descending to rank importance
            same_cat.sort(key=lambda x: x["val"], reverse=True)
            diff_cat.sort(key=lambda x: x["val"], reverse=True)
            
            def fmt_cat_peers(lst):
                if not lst: return ""
                names = [f"{x['name']} ({x['cat']} at {x['str']})" for x in lst]
                if len(names) > 5: return ", ".join(names[:5]) + f", and {len(names)-5} others"
                if len(names) == 1: return names[0]
                if len(names) == 2: return f"{names[0]} and {names[1]}"
                return ", ".join(names[:-1]) + f", and {names[-1]}"
                    
            m_clean = metric_name.lower().replace("percentage ", "").replace("percent ", "")
            draft = f"{fn}'s primary {m_clean} is {f_cat} ({f_str}). "
            
            same_str = fmt_cat_peers(same_cat)
            diff_str = fmt_cat_peers(diff_cat)
            
            if same_str and diff_str:
                draft += f"It shares this primary category with {same_str}, but differs from {diff_str}. "
            elif same_str:
                draft += f"It shares this primary category with {same_str}. "
            elif diff_str:
                draft += f"In contrast, the primary category differs for {diff_str}. "
                
            if peer_comb_cat:
                if peer_comb_cat == f_cat:
                    draft += f"Overall, its combined peers also share {f_cat} as the primary category."
                else:
                    draft += f"Overall, its combined peers are primarily driven by {peer_comb_cat}."
                    
            return draft.strip(), debug_math
        except Exception as e:
            return "N/A", f"Error: {str(e)}"

    def generate_peer_detailed_insight(self, fv_raw, peer_details, peer_avg_raw, fn, metric_name, is_percentage=False, m_type=""):
        """Deterministically calculates numerical relationships to all individual peers."""
        if not peer_details or fv_raw is None: return "N/A", "Missing Data"
        try:
            f = float(fv_raw)
            higher_than, lower_than, similar = [], [], []
            
            tolerance = max(abs(f * 0.02), 0.2) if is_percentage else max(abs(f * 0.02), 0.001)
            debug_math = f"Focus ({fn}): {f:,.4f} | Tol: +/-{tolerance:,.4f} | "
            f_str = self.format_value(f, metric_name, m_type)
            
            for p in peer_details:
                if fn.lower() == p["Name"].lower() or p["Name"].lower() in fn.lower(): continue
                p_val = float(p["Value"])
                p_str = self.format_value(p_val, metric_name, m_type)
                diff = f - p_val
                
                # Store formatted difference alongside each peer object
                if is_percentage:
                    diff_str = f"{abs(diff):.1f} pp"
                else:
                    diff_str = self.format_value(abs(diff), metric_name, m_type)
                peer_obj = {"name": p["Name"], "val": p_val, "str": p_str, "abs_diff": abs(diff), "diff_str": diff_str}
                debug_math += f"[{p['Name']} ({p_val:,.4f}): "
                
                if abs(diff) <= tolerance:
                    similar.append(peer_obj)
                    debug_math += "SIM] "
                elif diff > 0:
                    higher_than.append(peer_obj)
                    debug_math += f"Focus>Peer by {diff_str}] "
                else:
                    lower_than.append(peer_obj)
                    debug_math += f"Focus<Peer by {diff_str}] "
                    
            higher_than.sort(key=lambda x: x["abs_diff"], reverse=True)
            lower_than.sort(key=lambda x: x["abs_diff"], reverse=True)
            similar.sort(key=lambda x: x["abs_diff"], reverse=True)

            m_clean = metric_name.lower().replace("percentage ", "").replace("percent ", "")
            if is_percentage or "rate" in m_clean or "change" in m_clean:
                word_high, word_low = "higher", "lower"
            else:
                word_high, word_low = "larger", "smaller"
                
            draft = f"{fn}'s {m_clean} of {f_str} is "
            
            if peer_avg_raw is not None:
                p_avg = float(peer_avg_raw)
                p_avg_str = self.format_value(p_avg, metric_name, m_type)
                avg_diff = f - p_avg
                avg_diff_str = f"{abs(avg_diff):.1f} pp" if is_percentage else self.format_value(abs(avg_diff), metric_name, m_type)
                debug_math += f"| Peer Avg: {p_avg:,.4f} "
                if abs(f - p_avg) <= tolerance:
                    draft += f"nearly the same as its peer average of {p_avg_str}. "
                    debug_math += "-> Eval: Avg Similar."
                elif f > p_avg:
                    draft += f"{word_high} than its peer average of {p_avg_str} (by {avg_diff_str}). "
                    debug_math += f"-> Eval: Avg {word_high.capitalize()} by {avg_diff_str}."
                else:
                    draft += f"{word_low} than its peer average of {p_avg_str} (by {avg_diff_str}). "
                    debug_math += f"-> Eval: Avg {word_low.capitalize()} by {avg_diff_str}."
            else:
                draft += f"evaluated against individual peers. "

            def fmt_list(lst):
                """Format list with name, value, and delta."""
                if not lst: return ""
                names = [f"{x['name']} ({x['str']}, diff: {x['diff_str']})" for x in lst]
                if len(names) > 5: return ", ".join(names[:5]) + f", and {len(names)-5} others"
                if len(names) == 1: return names[0]
                if len(names) == 2: return f"{names[0]} and {names[1]}"
                return ", ".join(names[:-1]) + f", and {names[-1]}"

            parts = []
            if lower_than: parts.append(f"{word_low} than {fmt_list(lower_than)}")
            if higher_than: parts.append(f"{word_high} than {fmt_list(higher_than)}")
            if similar: parts.append(f"nearly the same as {fmt_list(similar)}")
            
            if parts: draft += "Specifically, it is " + ", while being ".join(parts) + "."
            else: draft += "It is not comparable to specific peers."
                
            return draft.strip(), debug_math
        except Exception as e:
            return "N/A", f"Error: {str(e)}"

    # ==========================================
    # 3. UNIFIED DATA ENGINE
    # ==========================================
    def combine_roles(self, df_result, value_col, metric_name, m_type, is_categorical=False, category_col=None, is_percentage_override=None):
        geo_data = {}
        if not is_categorical: df_result = df_result.sort(value_col) 
            
        role_dict = {}
        for r in df_result.filter(~pl.col("Role").is_in(["Peer", "Peer_Indiv"])).iter_rows(named=True):
            role = r["Role"]
            if role not in role_dict: role_dict[role] = []
            role_dict[role].append(r)
                
        for role, rows in role_dict.items():
            role_list = []
            for r in rows:
                if is_categorical: 
                    role_list.append({"Name": r["NAME"], "Formatted_Value": f"the '{r[category_col]}' group with {self.format_value(r[value_col], metric_name, m_type, is_percentage_override=is_percentage_override)}", "Raw_Value": r[value_col], "Category": r[category_col]})
                else: 
                    role_list.append({"Name": r["NAME"], "Formatted_Value": self.format_value(r[value_col], metric_name, m_type, is_percentage_override=is_percentage_override), "Raw_Value": r[value_col]})
            
            if role == "Focus":
                geo_data[role] = role_list[0]
            else:
                geo_data[role] = role_list
                
        df_peer = df_result.filter(pl.col("Role") == "Peer")
        if not df_peer.is_empty():
            if is_categorical:
                r = df_peer.row(0, named=True)
                geo_data["Peer"] = [{"Name": r["NAME"], "Formatted_Value": f"the '{r[category_col]}' group with {self.format_value(r[value_col], metric_name, m_type, is_percentage_override=is_percentage_override)}", "Raw_Value": r[value_col], "Category": r[category_col]}]
            else:
                peer_avg = df_peer[value_col].mean()
                geo_data["Peer"] = [{"Name": "Peers (Average)", "Formatted_Value": self.format_value(peer_avg, metric_name, m_type, is_percentage_override=is_percentage_override), "Raw_Value": peer_avg}]
                
                peer_details = []
                for r in df_peer.iter_rows(named=True):
                    peer_details.append({"Name": r["NAME"], "Value": r[value_col]})
                geo_data["Peer_Details"] = peer_details

        df_peer_indiv = df_result.filter(pl.col("Role") == "Peer_Indiv")
        if not df_peer_indiv.is_empty():
            peer_details = []
            for r in df_peer_indiv.iter_rows(named=True):
                cat = r[category_col] if is_categorical else None
                peer_details.append({"Name": r["NAME"], "Value": r[value_col], "Category": cat})
            geo_data["Peer_Details"] = peer_details
                
        return geo_data

    def calculate_metric_data(self, master_df, df_pyr, metric_name, variables_json, m_type, d_source=None, bp_curr="", bp_comp=""):
        m_lower = metric_name.lower()
        type_lower = str(m_type).lower()
        
        is_yoy_change = ("change" in m_lower and "cumulative" not in m_lower)
        is_percentage_metric = "percent" in type_lower or "rate" in type_lower or (("percent" in m_lower or "rate" in m_lower) and "numeric" not in type_lower)
        years_context = {"latest": "", "prev": ""}
        
        vars_dict = self.parse_variables(variables_json)
        value_types = []
        if "Value Type" in vars_dict:
            if isinstance(vars_dict["Value Type"], list):
                value_types = vars_dict["Value Type"]
            else:
                value_types = [vars_dict["Value Type"]]

        if self.is_population_share_metric(metric_name):
            return self.calculate_population_share(master_df, years_context, metric_name, m_type, is_change=False)
        if self.is_population_share_change_metric(metric_name):
            return self.calculate_population_share(master_df, years_context, metric_name, m_type, is_change=True)

        if "driver" in m_lower or "dynamics in drivers" in m_lower:
            is_cumulative = "cumulative" in m_lower
            if value_types:
                drivers = [vt.upper() for vt in value_types if isinstance(vt, str)]
            else:
                drivers = ["NATURALCHG", "DOMESTICMIG", "INTERNATIONALMIG"]
            df_d = master_df.filter(pl.col("Metric").is_in(drivers))
            
            if df_d.is_empty(): return None, years_context, True
            
            years = df_d["Year"].drop_nulls().unique().sort()
            if len(years) > 0: years_context["latest"] = str(years[-1])
            if len(years) > 1: years_context["prev"] = str(years[-2])
            
            if is_cumulative: df_d = df_d.filter(pl.col("Year") >= 1990)
            else: df_d = df_d.filter(pl.col("Year") == df_d["Year"].max())
            if df_d.is_empty(): return None, years_context, True
            
            df_std = df_d.filter(pl.col("Role") != "Peer")
            df_peer_indiv = df_d.filter(pl.col("Role") == "Peer")
            df_peer_comb = df_peer_indiv.with_columns(pl.lit("Peers (Combined)").alias("NAME"))
            df_combined = pl.concat([df_std, df_peer_comb, df_peer_indiv.with_columns(pl.lit("Peer_Indiv").alias("Role"))])
            
            agg = df_combined.group_by(["NAME", "Role", "Metric"]).agg(pl.col("Value").sum())
            largest = agg.with_columns(pl.col("Value").abs().alias("Abs_Val")).sort("Abs_Val", descending=True).group_by(["NAME", "Role"]).first()
            names = {"NATURALCHG": "Natural Change", "DOMESTICMIG": "Domestic Migration", "INTERNATIONALMIG": "International Migration"}
            largest = largest.with_columns(pl.col("Metric").replace(names).alias("Driver_Name"))
            
            return self.combine_roles(largest, "Value", metric_name, m_type, is_categorical=True, category_col="Driver_Name"), years_context, True

        elif "cumulative" in m_lower and "population" in m_lower and "change" in m_lower:
            df_pop = master_df.filter(pl.col("Metric") == "POPESTIMATE")
            if df_pop.is_empty() or df_pop.filter(pl.col("Year") == 1990).is_empty(): return None, years_context, False
            
            # Deduplicate df_pop to prevent cartesian product join duplicate rows
            df_pop = df_pop.group_by(["NAME", "Role", "Year"]).agg(pl.col("Value").mean())
            
            years = df_pop["Year"].drop_nulls().unique().sort()
            if len(years) > 0: years_context["latest"] = str(years[-1])
            if len(years) > 1: years_context["prev"] = str(years[-2])
            
            latest_year = df_pop["Year"].max()
            df_latest = df_pop.filter(pl.col("Year") == latest_year).select(["NAME", "Role", "Value"])
            df_1990 = df_pop.filter(pl.col("Year") == 1990).select(["NAME", "Role", "Value"])
            
            change_df = df_latest.join(df_1990, on=["NAME", "Role"], how="inner")
            
            if is_percentage_metric:
                change_df = change_df.with_columns((((pl.col("Value") - pl.col("Value_right")) / pl.col("Value_right")) * 100).alias("Change"))
            else:
                change_df = change_df.with_columns((pl.col("Value") - pl.col("Value_right")).alias("Change"))
                
            return self.combine_roles(change_df, "Change", metric_name, m_type), years_context, False

        elif "largest" in m_lower and ("age" in m_lower or "race" in m_lower or "gender" in m_lower or "housing" in m_lower) and "population_pyramid" in str(d_source).lower():
            years = df_pyr["YEAR"].drop_nulls().unique().sort()
            if len(years) == 0: return None, years_context, True
            
            latest = years[-1]
            years_context["latest"] = str(latest)
            if len(years) > 1: years_context["prev"] = str(years[-2])
            
            if "race" in m_lower:
                df_pyr_clean = df_pyr.with_columns(pl.col("Variable Group Description").str.replace(" Male", "").str.replace(" Female", "").alias("Group_Clean"))
                df_pyr_clean = df_pyr_clean.filter(~pl.col("Group_Clean").is_in(["Male", "Female", "Total", "All"]))
                group_col = "Group_Clean"
            else:
                df_pyr_clean = df_pyr.filter(pl.col("Age Group Description") != "All")
                group_col = "Age Group Description"
                
            df_std = df_pyr_clean.filter(pl.col("Role") != "Peer")
            df_peer_indiv = df_pyr_clean.filter(pl.col("Role") == "Peer")
            df_peer_comb = df_peer_indiv.with_columns(pl.lit("Peers (Combined)").alias("NAME"))
            df_combined = pl.concat([df_std, df_peer_comb, df_peer_indiv.with_columns(pl.lit("Peer_Indiv").alias("Role"))])

            if not is_yoy_change:
                df_l = df_combined.filter(pl.col("YEAR") == latest).group_by(["NAME", "Role", group_col]).agg(pl.col("Values").cast(pl.Float64).sum())
                largest = df_l.sort("Values", descending=True).group_by(["NAME", "Role"]).first()
                # FIX: If Focus is absent from the pyramid, return None so we skip cleanly
                # rather than letting the downstream fallback promote Broad to Focus.
                if largest.filter(pl.col("Role") == "Focus").is_empty():
                    print(f"  [Skipped] Metric '{metric_name}': 'Focus' geography not found in population pyramid data.")
                    return None, years_context, True
                return self.combine_roles(largest, "Values", metric_name, m_type, is_categorical=True, category_col=group_col), years_context, True
            else:
                if len(years) < 2: return None, years_context, True
                prev = years[-2] 
                
                df_l = df_combined.filter(pl.col("YEAR") == latest).group_by(["NAME", "Role", group_col]).agg(pl.col("Values").cast(pl.Float64).sum().alias("VL"))
                df_p = df_combined.filter(pl.col("YEAR") == prev).group_by(["NAME", "Role", group_col]).agg(pl.col("Values").cast(pl.Float64).sum().alias("VP"))
                change = df_l.join(df_p, on=["NAME", "Role", group_col], how="inner").with_columns((pl.col("VL") - pl.col("VP")).alias("Chg"))
                largest = change.sort("Chg", descending=True).group_by(["NAME", "Role"]).first()
                # FIX: Same guard for the YoY change variant
                if largest.filter(pl.col("Role") == "Focus").is_empty():
                    print(f"  [Skipped] Metric '{metric_name}': 'Focus' geography not found in population pyramid data.")
                    return None, years_context, True
                return self.combine_roles(largest, "Chg", metric_name, m_type, is_categorical=True, category_col=group_col), years_context, True

        data_source_lower = str(d_source).lower() if d_source is not None else ""
        variable_fields = []
        if "Fields" in vars_dict:
            if isinstance(vars_dict["Fields"], list):
                variable_fields = vars_dict["Fields"]
            elif isinstance(vars_dict["Fields"], str):
                variable_fields = [vars_dict["Fields"]]
            else:
                variable_fields = [str(vars_dict["Fields"])]
        elif "Label" in vars_dict:
            if isinstance(vars_dict["Label"], list):
                variable_fields = vars_dict["Label"]
            elif isinstance(vars_dict["Label"], str):
                variable_fields = [vars_dict["Label"]]
            else:
                variable_fields = [str(vars_dict["Label"])]
        elif "Value Type" in vars_dict:
            if isinstance(vars_dict["Value Type"], list):
                variable_fields = vars_dict["Value Type"]
            else:
                variable_fields = [vars_dict["Value Type"]]

        if "acs" in data_source_lower:
            if not variable_fields:
                return None, years_context, False

            # --- Multi-label categorical handler (e.g. Largest Age Group, Largest Race Group) ---
            # When the blueprint specifies multiple Label fields and the metric name implies
            # finding the largest/biggest category, we compare all labels and pick the winner.
            all_labels = []
            if "Label" in vars_dict:
                raw = vars_dict["Label"]
                all_labels = raw if isinstance(raw, list) else [raw]
            elif "Fields" in vars_dict:
                raw = vars_dict["Fields"]
                all_labels = raw if isinstance(raw, list) else [raw]

            is_largest_metric = "largest" in m_lower

            if is_largest_metric and len(all_labels) > 1:
                df_labels = master_df.filter(pl.col("Metric").is_in(all_labels))
                if df_labels.is_empty():
                    return None, years_context, False

                df_labels = self.normalize_percentage_metric_values(df_labels, "Value", metric_name, m_type)

                years = df_labels["Year"].drop_nulls().unique().sort()
                if len(years) == 0: return None, years_context, False
                if len(years) > 0: years_context["latest"] = str(years[-1])
                if len(years) > 1: years_context["prev"] = str(years[-2])
                latest_year = years[-1]

                if is_yoy_change:
                    if len(years) < 2: return None, years_context, False
                    prev_year = years[-2]
                    df_l = df_labels.filter(pl.col("Year") == latest_year).group_by(["NAME", "Role", "Metric"]).agg(pl.col("Value").mean())
                    df_p = df_labels.filter(pl.col("Year") == prev_year).group_by(["NAME", "Role", "Metric"]).agg(pl.col("Value").mean())
                    joined = df_l.join(df_p, on=["NAME", "Role", "Metric"], how="inner").with_columns(
                        (pl.col("Value") - pl.col("Value_right")).alias("Change")
                    )
                    # Pick the metric with the largest absolute change per geography
                    largest = joined.with_columns(pl.col("Change").abs().alias("Abs_Change")) \
                                    .sort("Abs_Change", descending=True) \
                                    .group_by(["NAME", "Role"]).first()
                    return self.combine_roles(largest, "Change", metric_name, m_type, is_categorical=True, category_col="Metric", is_percentage_override=True), years_context, True
                else:
                    df_latest = df_labels.filter(pl.col("Year") == latest_year).group_by(["NAME", "Role", "Metric"]).agg(pl.col("Value").mean())
                    # Pick the metric with the largest value per geography
                    largest = df_latest.sort("Value", descending=True).group_by(["NAME", "Role"]).first()
                    return self.combine_roles(largest, "Value", metric_name, m_type, is_categorical=True, category_col="Metric", is_percentage_override=True), years_context, True

            # --- Standard single/multi-field ACS handler ---
            if not variable_fields:
                return None, years_context, False
            df_metric = master_df.filter(pl.col("Metric").is_in(variable_fields))
            df_metric = self.prefer_pep_source(df_metric, metric_name)
            if df_metric.is_empty():
                return None, years_context, False

            # If there are multiple fields, sum them up per geography and year!
            if len(variable_fields) > 1:
                df_metric = df_metric.group_by(["NAME", "Role", "Year"]).agg(pl.col("Value").sum())

            df_metric = self.normalize_percentage_metric_values(df_metric, "Value", metric_name, m_type)

            years = df_metric["Year"].drop_nulls().unique().sort()
            if len(years) > 0: years_context["latest"] = str(years[-1])
            if len(years) > 1: years_context["prev"] = str(years[-2])
            latest_year = years[-1]

            is_cpi_source = "cpi" in data_source_lower
            
            # Robust monthly vs annual detection:
            # If the source is ACS, it's always annual (not monthly).
            # Otherwise, check if bp_curr is a 4-digit number (year) or contains year/annual.
            is_monthly = True
            if "acs" in data_source_lower:
                is_monthly = False
            else:
                bp_curr_str = str(bp_curr).lower().strip()
                if "year" in bp_curr_str or "annual" in bp_curr_str:
                    is_monthly = False
                else:
                    # Check if it's a 4-digit integer like 2024
                    try:
                        val = int(float(bp_curr_str))
                        if 1900 <= val <= 2100:
                            is_monthly = False
                    except ValueError:
                        pass

            if is_yoy_change:
                if len(years) < 2: return None, years_context, False
                prev_year = years[-2]
                df_l = df_metric.filter(pl.col("Year") == latest_year).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
                df_p = df_metric.filter(pl.col("Year") == prev_year).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())

                cpi_latest = self.get_cpi_value_for_period(master_df, latest_year, is_monthly)
                cpi_prev = self.get_cpi_value_for_period(master_df, prev_year, is_monthly)

                cpi_debug = {}
                if is_cpi_source and cpi_latest is not None and cpi_prev is not None and cpi_prev != 0:
                    ratio = cpi_latest / cpi_prev
                    df_p_adj = df_p.with_columns((pl.col("Value") * ratio).alias("Value"))
                    change_df = df_l.join(df_p_adj, on=["NAME", "Role"], how="inner").with_columns(
                        (pl.col("Value") - pl.col("Value_right")).alias("Change")
                    )
                    cpi_debug = {
                        "current_period": str(latest_year),
                        "comparison_period": str(prev_year),
                        "current_cpi": cpi_latest,
                        "comparison_cpi": cpi_prev,
                        "calculation": f"Value ({latest_year}) - (Value ({prev_year}) * ({cpi_latest} / {cpi_prev}))"
                    }
                else:
                    change_df = df_l.join(df_p, on=["NAME", "Role"], how="inner").with_columns(
                        (pl.col("Value") - pl.col("Value_right")).alias("Change")
                    )
                    cpi_debug = {
                        "current_period": str(latest_year),
                        "comparison_period": str(prev_year),
                        "current_cpi": cpi_latest if cpi_latest else "N/A",
                        "comparison_cpi": cpi_prev if cpi_prev else "N/A",
                        "calculation": f"Value ({latest_year}) - Value ({prev_year}) (no CPI adjustment)"
                    }

                change_df_raw = df_l.join(df_p, on=["NAME", "Role"], how="inner").with_columns(
                    (pl.col("Value") - pl.col("Value_right")).alias("Change")
                )

                geo_data_final = self.combine_roles(change_df, "Change", metric_name, m_type)
                geo_data_raw = self.combine_roles(change_df_raw, "Change", metric_name, m_type)
                geo_data_final["raw"] = geo_data_raw
                geo_data_final["cpi_debug_info"] = cpi_debug

                return geo_data_final, years_context, False
            else:
                df_dedup = df_metric.filter(pl.col("Year") == latest_year).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
                cpi_current = self.get_cpi_value_for_period(master_df, latest_year, is_monthly)

                # Keep single-year metrics in their nominal dollars for that specific year
                df_dedup_adj = df_dedup
                cpi_debug = {
                    "current_period": str(latest_year),
                    "comparison_period": "N/A",
                    "current_cpi": cpi_current if cpi_current else "N/A",
                    "comparison_cpi": "N/A",
                    "calculation": f"Value ({latest_year}) (Nominal - no CPI adjustment for single-year metric)"
                }

                geo_data_final = self.combine_roles(df_dedup_adj, "Value", metric_name, m_type)
                geo_data_raw = self.combine_roles(df_dedup, "Value", metric_name, m_type)
                geo_data_final["raw"] = geo_data_raw
                geo_data_final["cpi_debug_info"] = cpi_debug

                return geo_data_final, years_context, False

        if any(term in data_source_lower for term in ["hai-cpi table", "hai-cpi table-mortgagerates"]):
            if not variable_fields:
                return None, years_context, False

            numeric_fields = [f for f in variable_fields if f and f.lower() != "cpi"]
            if not numeric_fields:
                return None, years_context, False

            # Anchor periods to the HAI metrics (Listing Price, Rent, Income) which have local data.
            # National lookup metrics (Mortgage, CPI) should not drive the period selection.
            hai_anchor_metrics = ["Median Listing Price", "Calc-Median Monthly Rent", "Calc-Median HH Income", "Calc-Median Value of Owned Units"]
            df_anchor = master_df.filter(pl.col("Metric").is_in(hai_anchor_metrics) & (pl.col("Role") != "Benchmark"))
            
            if df_anchor.is_empty():
                # Fallback to the whole set if no anchor metrics found
                df_anchor = master_df.filter(pl.col("Metric").is_in(numeric_fields))
            
            available_months = df_anchor.select(pl.col("MonthKey")).drop_nulls().unique().sort("MonthKey").to_series().to_list()
            if not available_months:
                return None, years_context, False
            
            latest_month = available_months[-1]
            latest_year = latest_month // 100
            years_context["latest"] = str(latest_year)

            # Look for a month exactly 12 months ago
            prev_month = latest_month - 100
            if prev_month not in available_months:
                prev_month_candidates = [m for m in available_months if m < latest_month]
                prev_month = prev_month_candidates[-1] if prev_month_candidates else None
            
            if prev_month:
                years_context["prev"] = str(prev_month // 100)
            
            # Now filter the main df for processing
            df_hai_metric = master_df.filter(pl.col("Metric").is_in(numeric_fields))
            if df_hai_metric.is_empty():
                return None, years_context, False
                
            def process_mortgage(month_target):
                # Treated as a national lookup value (like CPI)
                m_val = self.get_mortgage_value(master_df, month_target, is_monthly=True)
                if m_val is None: return pl.DataFrame()
                
                # Filter for local metrics only (e.g. Median Listing Price)
                comp_df = df_hai_metric.filter((pl.col("MonthKey") == month_target) & (pl.col("Metric") == "Median Listing Price"))
                if comp_df.is_empty(): return pl.DataFrame()
                
                # Calculate payment using the national lookup rate broadcasted to local geos
                P = pl.col("Value") * 0.8
                r = (m_val / 100.0) / 12.0
                n = 360
                payment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
                return comp_df.with_columns(payment.alias("Value")).select(["NAME", "Role", "Value"])
                
            def process_hai(month_target):
                # Treated as a national lookup value (like CPI)
                m_val = self.get_mortgage_value(master_df, month_target, is_monthly=True)
                if m_val is None: return pl.DataFrame()
                
                # Filter for local metrics only
                comp_df = df_hai_metric.filter(pl.col("MonthKey") == month_target).filter(pl.col("Metric").is_in(["Median Listing Price", "Calc-Median HH Income"]))
                if comp_df.is_empty(): return pl.DataFrame()
                
                # Pivot local metrics
                df_pivot = comp_df.pivot(values="Value", index=["NAME", "Role"], columns="Metric", aggregate_function="mean")
                if "Median Listing Price" not in df_pivot.columns or "Calc-Median HH Income" not in df_pivot.columns:
                    return pl.DataFrame()
                    
                # Calculate HAI using the national lookup rate broadcasted to local geos
                P = pl.col("Median Listing Price") * 0.8
                r = (m_val / 100.0) / 12.0
                n = 360
                payment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
                q_inc = payment * 12 * 4
                hai = (pl.col("Calc-Median HH Income") / q_inc) * 100
                return df_pivot.with_columns(hai.alias("Value")).select(["NAME", "Role", "Value"])
                
            is_cpi_source = "cpi" in data_source_lower
            
            # Robust monthly vs annual detection:
            # If the source is ACS, it's always annual (not monthly).
            # Otherwise, check if bp_curr is a 4-digit number (year) or contains year/annual.
            is_monthly = True
            if "acs" in data_source_lower:
                is_monthly = False
            else:
                bp_curr_str = str(bp_curr).lower().strip()
                if "year" in bp_curr_str or "annual" in bp_curr_str:
                    is_monthly = False
                else:
                    # Check if it's a 4-digit integer like 2024
                    try:
                        val = int(float(bp_curr_str))
                        if 1900 <= val <= 2100:
                            is_monthly = False
                    except ValueError:
                        pass

            # If standard relative 12-month change adjusted by CPI
            if "12-month change" in m_lower or ("change" in m_lower and "adjusted by the cpi" in m_lower):
                if prev_month is None:
                    return None, years_context, False
                
                if "mortgage payment to median monthly rent" in m_lower:
                    mort_latest = process_mortgage(latest_month)
                    mort_prev = process_mortgage(prev_month)
                    rent_latest = df_hai_metric.filter(pl.col("Metric") == "Calc-Median Monthly Rent").drop_nulls("Year").sort("Year").group_by(["NAME", "Role"]).last()
                    rent_prev = rent_latest # Fallback to latest ACS year if prior isn't dynamically matched
                    if mort_latest.is_empty() or mort_prev.is_empty() or rent_latest.is_empty(): return None, years_context, False
                    
                    diff_latest = mort_latest.join(rent_latest, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Value")).select(["NAME", "Role", "Value"])
                    diff_prev = mort_prev.join(rent_prev, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Value")).select(["NAME", "Role", "Value"])
                    change_df = diff_latest.join(diff_prev, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Change"))
                    
                    geo_data_final = self.combine_roles(change_df, "Change", metric_name, m_type)
                    geo_data_final["raw"] = geo_data_final
                    geo_data_final["cpi_debug_info"] = {"calculation": "Mortgage payment to rent difference change (no CPI adjustment)"}
                    return geo_data_final, years_context, False
                
                elif "median list price to median value" in m_lower and "comparison" in m_lower:
                    list_latest = df_hai_metric.filter((pl.col("MonthKey") == latest_month) & (pl.col("Metric") == "Median Listing Price")).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
                    val_latest = df_hai_metric.filter(pl.col("Metric") == "Calc-Median Value of Owned Units").drop_nulls("Year").sort("Year").group_by(["NAME", "Role"]).last()
                    if list_latest.is_empty() or val_latest.is_empty(): return None, years_context, False
                    
                    list_prev = df_hai_metric.filter((pl.col("MonthKey") == prev_month) & (pl.col("Metric") == "Median Listing Price")).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
                    val_prev = val_latest # Fallback to latest ACS year if prior isn't dynamically matched
                    if list_prev.is_empty() or val_prev.is_empty(): return None, years_context, False
                    
                    diff_latest = list_latest.join(val_latest, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Value")).select(["NAME", "Role", "Value"])
                    diff_prev = list_prev.join(val_prev, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Value")).select(["NAME", "Role", "Value"])
                    change_df = diff_latest.join(diff_prev, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Change"))
                    
                    geo_data_final = self.combine_roles(change_df, "Change", metric_name, m_type)
                    geo_data_final["raw"] = geo_data_final
                    geo_data_final["cpi_debug_info"] = {"calculation": "List price to value difference change (no CPI adjustment)"}
                    return geo_data_final, years_context, False
                
                elif "housing affordability index" in m_lower:
                    df_latest = process_hai(latest_month)
                    df_prev = process_hai(prev_month)
                elif "estimated mortgage payment" in m_lower:
                    df_latest = process_mortgage(latest_month)
                    df_prev = process_mortgage(prev_month)
                else:
                    field = numeric_fields[0]
                    df_latest = df_hai_metric.filter((pl.col("MonthKey") == latest_month) & (pl.col("Metric") == field)).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
                    df_prev = df_hai_metric.filter((pl.col("MonthKey") == prev_month) & (pl.col("Metric") == field)).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())

                if 'df_latest' not in locals() or 'df_prev' not in locals() or df_latest.is_empty() or df_prev.is_empty():
                    return None, years_context, False

                cpi_debug = {}
                df_latest_adj = df_latest
                df_prev_adj = df_prev
                
                curr_period = latest_month if is_monthly else int(latest_month // 100)
                prev_period = prev_month if is_monthly else int(prev_month // 100)
                
                cpi_latest_val, cpi_latest_p = self.get_latest_cpi_value(master_df, is_monthly)
                cpi_curr_val = self.get_cpi_value_for_period(master_df, curr_period, is_monthly)
                cpi_prev_val = self.get_cpi_value_for_period(master_df, prev_period, is_monthly)
                
                if is_cpi_source and cpi_latest_val is not None and cpi_curr_val is not None and cpi_prev_val is not None and cpi_curr_val != 0 and cpi_prev_val != 0:
                    df_latest_adj = df_latest.with_columns((pl.col("Value") * (cpi_latest_val / cpi_curr_val)).alias("Value"))
                    df_prev_adj = df_prev.with_columns((pl.col("Value") * (cpi_latest_val / cpi_prev_val)).alias("Value"))
                    cpi_debug = {
                        "current_period": str(curr_period),
                        "comparison_period": str(prev_period),
                        "current_cpi": cpi_curr_val,
                        "comparison_cpi": cpi_prev_val,
                        "calculation": f"Value ({curr_period}) * ({cpi_latest_val} / {cpi_curr_val}) - Value ({prev_period}) * ({cpi_latest_val} / {cpi_prev_val})"
                    }
                else:
                    cpi_debug = {
                        "current_period": str(curr_period),
                        "comparison_period": str(prev_period),
                        "current_cpi": cpi_curr_val if cpi_curr_val else "N/A",
                        "comparison_cpi": cpi_prev_val if cpi_prev_val else "N/A",
                        "calculation": f"Value ({curr_period}) - Value ({prev_period}) (no CPI adjustment)"
                    }

                change_df_adjusted = df_latest_adj.join(df_prev_adj, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Change"))
                change_df_raw = df_latest.join(df_prev, on=["NAME", "Role"], how="inner").with_columns((pl.col("Value") - pl.col("Value_right")).alias("Change"))
                
                geo_data_final = self.combine_roles(change_df_adjusted, "Change", metric_name, m_type)
                geo_data_raw = self.combine_roles(change_df_raw, "Change", metric_name, m_type)
                geo_data_final["raw"] = geo_data_raw
                geo_data_final["cpi_debug_info"] = cpi_debug
                
                return geo_data_final, years_context, False

            if "compare estimated mortgage payment to median monthly rent" in m_lower:
                df_mort = process_mortgage(latest_month)
                rent_df = df_hai_metric.filter(pl.col("Metric") == "Calc-Median Monthly Rent").drop_nulls("Year").sort("Year").group_by(["NAME", "Role"]).last()
                if rent_df.is_empty() or df_mort.is_empty(): return None, years_context, False
                
                df_mort = df_mort.rename({"Value": "Estimated mortgage payment"}).select(["NAME", "Role", "Estimated mortgage payment"])
                rent_df = rent_df.rename({"Value": "Calc-Median Monthly Rent"}).select(["NAME", "Role", "Calc-Median Monthly Rent"])
                
                comp_df = df_mort.join(rent_df, on=["NAME", "Role"], how="inner")
                max_rows = comp_df.unpivot(index=["NAME", "Role"], variable_name="Metric", value_name="Value").sort("Value", descending=True).group_by(["NAME", "Role"]).first()
                return self.combine_roles(max_rows, "Value", metric_name, m_type, is_categorical=True, category_col="Metric"), years_context, True

            if "housing affordability index" in m_lower:
                df_hai = process_hai(latest_month)
                if df_hai.is_empty(): return None, years_context, False
                return self.combine_roles(df_hai, "Value", metric_name, m_type), years_context, False

            if "estimated mortgage payment" in m_lower:
                df_mort = process_mortgage(latest_month)
                if df_mort.is_empty(): return None, years_context, False
                return self.combine_roles(df_mort, "Value", metric_name, m_type), years_context, False
                
            if "compare median list price to median value" in m_lower:
                list_price = df_hai_metric.filter((pl.col("MonthKey") == latest_month) & (pl.col("Metric") == "Median Listing Price")).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
                value_units = df_hai_metric.filter(pl.col("Metric") == "Calc-Median Value of Owned Units").drop_nulls("Year").sort("Year").group_by(["NAME", "Role"]).last()
                if list_price.is_empty() or value_units.is_empty(): return None, years_context, False
                
                list_price = list_price.with_columns(pl.lit("Median Listing Price").alias("Metric")).select(["NAME", "Role", "Metric", "Value"])
                value_units = value_units.with_columns(pl.lit("Calc-Median Value of Owned Units").alias("Metric")).select(["NAME", "Role", "Metric", "Value"])
                
                comp_df = pl.concat([list_price, value_units])
                max_rows = comp_df.sort("Value", descending=True).group_by(["NAME", "Role"]).first()
                return self.combine_roles(max_rows, "Value", metric_name, m_type, is_categorical=True, category_col="Metric"), years_context, True

            if "median" in m_lower or "compare" in m_lower:
                df_latest = df_hai_metric.filter(pl.col("MonthKey") == latest_month).group_by(["NAME", "Role", "Metric"]).agg(pl.col("Value").mean())
                
                curr_period = latest_month if is_monthly else int(latest_month // 100)
                cpi_curr_val = self.get_cpi_value_for_period(master_df, curr_period, is_monthly)
                
                # Keep single-point metrics nominal
                df_latest_adj = df_latest
                cpi_debug = {
                    "current_period": str(curr_period),
                    "comparison_period": "N/A",
                    "current_cpi": cpi_curr_val if cpi_curr_val else "N/A",
                    "comparison_cpi": "N/A",
                    "calculation": f"Value ({curr_period}) (Nominal - no CPI adjustment for single-period metric)"
                }
                
                result_dict = self.combine_roles(df_latest_adj, "Value", metric_name, m_type)
                result_dict_raw = self.combine_roles(df_latest, "Value", metric_name, m_type)
                result_dict["raw"] = result_dict_raw
                result_dict["cpi_debug_info"] = cpi_debug
            else:
                result_dict = None

            if result_dict:
                # Apply Lookup Bypass for HAI sources if needed
                if ("mortgagerates" in data_source_lower or "cpi" in data_source_lower):
                    if 'Focus' not in result_dict and 'Benchmark' in result_dict and len(result_dict['Benchmark']) > 0:
                        result_dict['Focus'] = result_dict['Benchmark'][0]
                return result_dict, years_context, False

            return None, years_context, False

        target_raw = metric_name
        if value_types:
            df_metric = master_df.filter(pl.col("Metric").is_in([vt for vt in value_types if isinstance(vt, str)]))
        else:
            df_metric = master_df.filter(pl.col("Metric") == target_raw)
        df_metric = self.prefer_pep_source(df_metric, metric_name)
        if df_metric.is_empty():
            return None, years_context, False

        df_metric = self.normalize_percentage_metric_values(df_metric, "Value", metric_name, m_type)

        years = df_metric["Year"].drop_nulls().unique().sort()
        if len(years) > 0: years_context["latest"] = str(years[-1])
        if len(years) > 1: years_context["prev"] = str(years[-2])

        latest_year = df_metric["Year"].max()

        if is_yoy_change:
            if len(years) < 2:
                return None, years_context, False
            prev_year = years[-2]

            df_l_dedup = df_metric.filter(pl.col("Year") == latest_year).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
            df_p_dedup = df_metric.filter(pl.col("Year") == prev_year).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
            change_df = df_l_dedup.join(df_p_dedup, on=["NAME", "Role"], how="inner")

            max_val = change_df["Value_right"].max()
            if is_percentage_metric and max_val and max_val > 100:
                change_df = change_df.with_columns((((pl.col("Value") - pl.col("Value_right")) / pl.col("Value_right")) * 100).alias("Change"))
            else:
                change_df = change_df.with_columns((pl.col("Value") - pl.col("Value_right")).alias("Change"))

            result_df = self.combine_roles(change_df, "Change", metric_name, m_type)
        else:
            df_dedup = df_metric.filter(pl.col("Year") == latest_year).group_by(["NAME", "Role"]).agg(pl.col("Value").mean())
            result_df = self.combine_roles(df_dedup, "Value", metric_name, m_type)

        # --- LOOKUP BYPASS ---
        # If the source is a national lookup (Mortgage Rates or CPI) and no local Focus exists,
        # promote the Benchmark (National) data to Focus so the metric is not skipped.
        data_src_lower = str(d_source).lower() if d_source else ""
        if ("mortgagerates" in data_src_lower or "cpi" in data_src_lower) and result_df:
            if 'Focus' not in result_df and 'Benchmark' in result_df and len(result_df['Benchmark']) > 0:
                result_df['Focus'] = result_df['Benchmark'][0]

        return result_df, years_context, False

    # ==========================================
    # 4. LLM INTERFACE & SYNTHESIS
    # ==========================================
    def generate_text(self, prompt: str, as_json: bool = False, json_key: str = "overall_insight", max_words: int = 150) -> str:
        """
        Generates text using the configured model (Gemini or Ollama) with temperature 0.1.
        Includes retry limits/fallbacks and formats/cleans the response as JSON if requested.
        """
        if not prompt:
            return "N/A"

        limits = [800, 1500, 2500]
        last_error = "Unknown Error"

        for limit in limits:
            try:
                if self.mode == "gemini":
                    config_args = {"temperature": 0.1}
                    if as_json:
                        config_args["response_mime_type"] = "application/json"

                    response = self.gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_args)
                    )
                    resp = response.text.strip()
                else:
                    resp = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": 0.1, "seed": random.randint(1, 100000), "num_predict": limit}
                    )["response"].strip()

                resp_clean = re.sub(r'```json\s*', '', resp, flags=re.IGNORECASE)
                resp_clean = re.sub(r'```\s*', '', resp_clean)

                if as_json:
                    match = re.search(r'\{.*?\}', resp_clean, re.DOTALL)
                    if match:
                        try:
                            val = json.loads(match.group(0)).get(json_key, "")
                            if not val:
                                val = match.group(0)
                        except Exception:
                            val = match.group(0)
                    else:
                        val = resp_clean

                    val = re.sub(r'\{?\s*\"?' + json_key + r'\"?\s*:\s*\"?', '', val, flags=re.IGNORECASE)
                    val = val.replace('"}', '').replace('}', '').strip()
                    val = re.sub(r'^\"', '', val)
                    val = re.sub(r'\"$', '', val)
                    val = re.sub(r"^.*?(revised_sentence|overall_insight|insight|topic_summary|complete_summary)\"?\s*:\s*\"?", "", val, flags=re.IGNORECASE).strip()
                    val = re.sub(r"^.*?(summary|sentence|data|revised|professional).*?:", "", val, flags=re.IGNORECASE).strip()

                    if not val:
                        last_error = "Empty parsed value"
                        continue

                    if not re.search(r'[.!?\"\'\}]$', val) and limit != limits[-1] and self.mode == "ollama":
                        last_error = "Truncated text detected"
                        continue

                    val = self.sanitize_text(val.strip())
                    return self.enforce_word_limit(val, max_words)
                else:
                    resp_clean = self.sanitize_text(resp_clean.strip())
                    return self.enforce_word_limit(resp_clean, max_words)

            except Exception as e:
                last_error = str(e)
                continue
        return f"Error: {last_error}"

    def sanitize_text(self, text: str) -> str:
        """Replace common smart/curly Unicode characters with clean ASCII equivalents."""
        if not isinstance(text, str): return text
        replacements = {
            '\u2018': "'", '\u2019': "'",   # left/right single curly quotes -> straight apostrophe
            '\u201c': '"', '\u201d': '"',   # left/right double curly quotes -> straight double quote
            '\u2013': '-',                   # en dash -> hyphen
            '\u2014': '--',                  # em dash -> double hyphen
            '\u2026': '...',                 # ellipsis -> three dots
            '\u00a0': ' ',                   # non-breaking space -> regular space
            '\ufffd': '',                    # Unicode replacement character -> remove
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        # Catch any remaining mojibake from misread UTF-8 (e.g. â€™ pattern)
        try:
            text = text.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        # Final pass: strip any remaining non-ASCII chars that slipped through
        text = text.encode('ascii', errors='replace').decode('ascii').replace('?', "'")
        return text

    def enforce_word_limit(self, text: str, max_words: int = 150) -> str:
        if not isinstance(text, str) or max_words is None:
            return text
        words = text.strip().split()
        if len(words) <= max_words:
            return text.strip()
        truncated = " ".join(words[:max_words]).strip()
        if truncated and truncated[-1] not in ".!?":
            truncated = truncated.rstrip(' ,;:-') + '.'
        return truncated

    def parse_variables(self, variables_json: str) -> dict:
        if not isinstance(variables_json, str) or not variables_json.strip():
            return {}

        cleaned = variables_json.strip()
        cleaned = cleaned.replace("''", '"').replace("‘", '"').replace("’", '"').replace('“', '"').replace('”', '"')

        # Try valid JSON first (only if there are no duplicate keys, as json.loads silently discards them)
        try:
            keys = re.findall(r'"([^"]+)"\s*:', cleaned)
            if len(keys) == len(set(keys)):
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"Fields": parsed}
        except Exception:
            pass

        # Handle JSON-like lists without explicit keys: {"A", "B"}
        list_values = re.findall(r'"([^"]+)"', cleaned)
        if list_values and ":" not in cleaned:
            return {"Fields": list_values}

        # Handle simple key:value pairs in plain text
        matches = re.findall(r'"([^\"]+)"\s*:\s*"([^\"]+)"', cleaned)
        parsed = {}
        for key, value in matches:
            if key in parsed:
                if isinstance(parsed[key], list):
                    parsed[key].append(value)
                else:
                    parsed[key] = [parsed[key], value]
            else:
                parsed[key] = value

        return parsed

    def get_cpi_value_for_period(self, master_df: pl.DataFrame, period: int, is_monthly: bool):
        if period is None: return None
        if is_monthly:
            row = master_df.filter((pl.col("Metric") == "CPI (Monthly)") & (pl.col("MonthKey") == period))
        else:
            row = master_df.filter((pl.col("Metric") == "Average CPI (Annual)") & (pl.col("Year") == period))
        if row.is_empty(): return None
        return float(row.select(pl.col("Value").mean()).item())

    def get_latest_cpi_value(self, master_df: pl.DataFrame, is_monthly: bool):
        if is_monthly:
            rows = master_df.filter(pl.col("Metric") == "CPI (Monthly)").drop_nulls(subset=["MonthKey"]).sort("MonthKey", descending=True)
            if rows.is_empty(): return None, None
            val = float(rows.select(pl.col("Value")).head(1).item())
            p = int(rows.select(pl.col("MonthKey")).head(1).item())
            return val, p
        else:
            rows = master_df.filter(pl.col("Metric") == "Average CPI (Annual)").drop_nulls(subset=["Year"]).sort("Year", descending=True)
            if rows.is_empty(): return None, None
            val = float(rows.select(pl.col("Value")).head(1).item())
            p = int(rows.select(pl.col("Year")).head(1).item())
            return val, p

    def get_cpi_value(self, master_df: pl.DataFrame, period: int, is_monthly: bool = False):
        return self.get_cpi_value_for_period(master_df, period, is_monthly)

    def get_latest_cpi(self, master_df: pl.DataFrame, is_monthly: bool = False):
        val, _ = self.get_latest_cpi_value(master_df, is_monthly)
        return val

    def get_average_cpi(self, master_df: pl.DataFrame, year: int):
        return self.get_cpi_value(master_df, year, is_monthly=False)

    def get_cpi_ratio(self, master_df: pl.DataFrame, from_year: int, to_year: int):
        cpi_from = self.get_average_cpi(master_df, from_year)
        cpi_to = self.get_average_cpi(master_df, to_year)
        if cpi_from and cpi_to and cpi_from != 0:
            return cpi_to / cpi_from
        return None

    def get_mortgage_value(self, master_df: pl.DataFrame, period: int, is_monthly: bool = True):
        """Fetches the national 30-year fixed mortgage rate for a given period."""
        if period is None: return None
        if is_monthly:
            row = master_df.filter((pl.col("Metric") == "Monthly Avg 30yr") & (pl.col("MonthKey") == period))
        else:
            row = master_df.filter((pl.col("Metric") == "Monthly Avg 30yr") & (pl.col("Year") == period))
        if row.is_empty(): return None
        return float(row.select(pl.col("Value").mean()).item())

    def get_latest_mortgage_value(self, master_df: pl.DataFrame, is_monthly: bool = True):
        """Fetches the most recent national 30-year fixed mortgage rate."""
        if is_monthly:
            rows = master_df.filter(pl.col("Metric") == "Monthly Avg 30yr").drop_nulls(subset=["MonthKey"]).sort("MonthKey", descending=True)
        else:
            rows = master_df.filter(pl.col("Metric") == "Monthly Avg 30yr").drop_nulls(subset=["Year"]).sort("Year", descending=True)
        if rows.is_empty(): return None
        return float(rows.select(pl.col("Value")).head(1).item())

    def adjust_to_current_dollars(self, value, from_year: int, to_year: int, master_df: pl.DataFrame):
        if value is None:
            return None
        try:
            ratio = self.get_cpi_ratio(master_df, from_year, to_year)
            if ratio is None:
                return float(value)
            return float(value) * ratio
        except Exception:
            return float(value)

    def prefer_pep_source(self, df: pl.DataFrame, metric_name: str) -> pl.DataFrame:
        if df.is_empty():
            return df
        if not isinstance(metric_name, str):
            return df
        metric_lower = metric_name.lower()
        if "population" not in metric_lower and "pop" not in metric_lower:
            return df
        pep_filters = []
        if "Source" in df.columns:
            pep_filters.append(pl.col("Source").str.to_lowercase().str.contains("pep"))
        if "Dataset" in df.columns:
            pep_filters.append(pl.col("Dataset").str.to_lowercase().str.contains("pep"))
        if not pep_filters:
            return df
        combined_filter = pep_filters[0]
        if len(pep_filters) > 1:
            combined_filter = combined_filter | pep_filters[1]
        df_pep = df.filter(combined_filter)
        return df_pep if not df_pep.is_empty() else df

    def is_population_share_metric(self, metric_name: str) -> bool:
        if not isinstance(metric_name, str):
            return False
        lower = metric_name.lower()
        return "share" in lower and "broad region" in lower and "change" not in lower

    def is_population_share_change_metric(self, metric_name: str) -> bool:
        if not isinstance(metric_name, str):
            return False
        lower = metric_name.lower()
        return "change" in lower and "share" in lower and "broad region" in lower

    def calculate_population_share(self, master_df: pl.DataFrame, years_context: dict, metric_name: str, m_type: str, is_change: bool):
        df_pop = master_df.filter((pl.col("Metric") == "POPESTIMATE") & pl.col("Role").is_in(["Focus", "Broad"]))
        
        has_focus = not df_pop.filter(pl.col("Role") == "Focus").is_empty()
        has_broad = not df_pop.filter(pl.col("Role") == "Broad").is_empty()
        
        if not (has_focus and has_broad):
            df_pop = master_df.filter((pl.col("Metric") == "Population") & pl.col("Role").is_in(["Focus", "Broad"]))
            
        if df_pop.is_empty():
            return None, years_context, False

        focus_name_df = df_pop.filter(pl.col("Role") == "Focus").select("NAME").unique()
        if focus_name_df.is_empty():
            return None, years_context, False
        focus_geo_name = focus_name_df["NAME"][0]

        years = df_pop["Year"].drop_nulls().unique().sort()
        if len(years) > 0:
            years_context["latest"] = str(years[-1])
        if len(years) > 1:
            years_context["prev"] = str(years[-2])

        if is_change and len(years) < 2:
            return None, years_context, False

        year_role = df_pop.group_by(["Year", "Role"]).agg(pl.col("Value").mean())
        focus = year_role.filter(pl.col("Role") == "Focus").rename({"Value": "Focus_Value"})
        broad = year_role.filter(pl.col("Role") == "Broad").rename({"Value": "Broad_Value"})
        joined = focus.join(broad, on="Year", how="inner").with_columns((pl.col("Focus_Value") / pl.col("Broad_Value") * 100).alias("Share"))
        if joined.is_empty():
            return None, years_context, False

        if is_change:
            joined = joined.sort("Year")
            latest_row = joined[-1].to_dicts()[0]
            previous_row = joined[-2].to_dicts()[0]
            change = float(latest_row["Share"] - previous_row["Share"])
            df_change = pl.DataFrame({"NAME": [focus_geo_name], "Role": ["Focus"], "Metric": [metric_name], "Change": [change]})
            return self.combine_roles(df_change, "Change", metric_name, m_type), years_context, False

        latest_share = joined.filter(pl.col("Year") == years[-1]).select([pl.col("Share")])
        if latest_share.is_empty():
            return None, years_context, False

        value = float(latest_share.row(0)[0])
        df_share = pl.DataFrame({"NAME": [focus_geo_name], "Role": ["Focus"], "Metric": [metric_name], "Value": [value]})
        return self.combine_roles(df_share, "Value", metric_name, m_type), years_context, False

    def normalize_percentage_metric_values(self, df: pl.DataFrame, value_col: str, metric_name: str, m_type: str) -> pl.DataFrame:
        """Scale proportion-style values to percentages for percentage metrics when appropriate."""
        if df.is_empty():
            return df

        m_lower = str(metric_name).lower()
        type_lower = str(m_type).lower()
        is_percentage_metric = (
            "percent" in type_lower or 
            "rate" in type_lower or 
            "categorical" in type_lower or
            "largest" in m_lower or
            (("percent" in m_lower or "rate" in m_lower) and "numeric" not in type_lower)
        )
        if not is_percentage_metric:
            return df

        abs_max = df.select(pl.col(value_col).abs().max()).item()
        if abs_max is not None and abs_max <= 1.05 and abs_max > 0:
            return df.with_columns(pl.col(value_col) * 100)
        return df

    def enforce_sentence_capitalization(self, text: str) -> str:
        """Ensure each sentence starts with a capital letter while preserving existing interior casing."""
        if not isinstance(text, str):
            return text

        text = text.strip()
        if not text:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        corrected = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]
            corrected.append(sentence)
        return " ".join(corrected)

    def apply_grammar_layer(self, text: str) -> str:
        """Kept for backward compatibility; delegates to apply_capitalization_layer with no geo list."""
        return self.apply_capitalization_layer(text, [])

    def apply_geography_standardization_layer(self, text: str, valid_geos: list) -> str:
        """Kept for backward compatibility; delegates to apply_capitalization_layer."""
        return self.apply_capitalization_layer(text, valid_geos)

    def apply_capitalization_layer(self, text: str, valid_geos: list) -> str:
        """Single unified pass: fix grammar/capitalization AND geography name casing in one LLM call."""
        if text in ["N/A", "JSON Error", "JSON Parse Error", "JSON Format Error", ""] or str(text).startswith("Error:"):
            return text
        
        valid_geos_clean = list(set([
            g.replace("Peers (Average)", "its peer average").replace("Peers (Combined)", "its combined peers")
            for g in valid_geos if g
        ]))
        geo_str = ", ".join([f'"{g}"' for g in valid_geos_clean]) if valid_geos_clean else "none provided"
        
        p = f"""You are a strict copyeditor. Apply the following rules to the sentence below and output only the corrected sentence.

RULES (apply ALL of them):
1. GEOGRAPHY CASING - use Title Case for all place names:
   - Every word in a city, county, region, or state name gets a capital first letter.
   - Examples of correct format: "Fife, WA" not "FIFE, WA"; "Pierce County" not "PIERCE COUNTY";
     "Seattle MSA" not "SEATTLE MSA"; "Puget Sound Region" not "PUGET SOUND REGION";
     "Washington" not "WASHINGTON"; "United States" not "UNITED STATES".
   - Authoritative list of valid geography names for this sentence: {geo_str}.
   - Match each geography in the sentence to the closest name in that list and use that exact casing.
2. CATEGORY / DESCRIPTOR CASING - lowercase unless starting a sentence:
   - Demographic categories, age groups, race/ethnicity labels, and drivers of change are all lowercase.
   - Examples: "natural change", "domestic migration", "45-54 age group", "female", "white alone".
   - CRITICAL EXCEPTION: If any category or descriptor is the first word of a sentence, its first letter MUST be capitalized (e.g., "International migration is..." instead of "international migration is...").
3. SENTENCE STARTS - the first letter of every sentence must be capitalized. This rule takes absolute priority over Rule 2. The first letter of any sentence must be capitalized, even if it is a demographic term, age group, or driver of change.
4. GRAMMAR - fix any obvious grammatical errors, but do NOT change wording, facts, or numbers.
5. Do NOT add, remove, or paraphrase any information.

Original Sentence: {text}

Output STRICTLY as a valid JSON object with no extra text:
{{ "revised_sentence": "corrected sentence here" }}
"""
        revised = self.generate_text(p, as_json=True, json_key="revised_sentence", max_words=150)
        return self.enforce_sentence_capitalization(revised)

    # ==========================================
    # 5. PIPELINE EXECUTION
    # ==========================================
    def run(self, blueprint_path: str, acs_path: str, components_path: str, pyramid_path: str,
            hai_path: str = "examples/HAI.csv", cpi_path: str = "examples/cpi.xlsx", mortgage_path: str = "examples/MortgageRates.csv",
            sheet_name: str = 'AI Summary', output_path: str = 'AI_Summary.csv') -> pd.DataFrame:
        """
        Executes the AI Summary generation pipeline.

        Args:
            blueprint_path (str): Path to Excel blueprint (e.g. "Metric Topics (DRAFT).xlsx").
            acs_path (str): Path to ACS series CSV.
            components_path (str): Path to components of change CSV.
            pyramid_path (str): Path to population pyramid CSV.
            hai_path (str): Path to HAI monthly dataset CSV.
            cpi_path (str): Path to CPI lookup workbook.
            sheet_name (str): Excel sheet name to use (default: 'AI Summary').
            output_path (str): Path to save the final CSV output.

        Returns:
            pd.DataFrame: DataFrame containing the final synthesized results.
        """
        print("Loading data and Blueprint...")
        df_acs = pl.read_csv(acs_path, ignore_errors=True)
        df_comp = pl.read_csv(components_path, ignore_errors=True)
        df_pyr = pl.read_csv(pyramid_path, ignore_errors=True)
        df_hai = pl.DataFrame()
        df_cpi = pl.DataFrame()
        df_mortgage = pl.DataFrame()
        
        try:
            # Load Excel blueprint using pandas (handles locked/open sheets better on Windows)
            # and convert to Polars.
            pd_blueprint = pd.read_excel(blueprint_path, sheet_name=sheet_name)
            
            # Ensure all column names are strings
            pd_blueprint.columns = [str(col).strip() for col in pd_blueprint.columns]
            
            # Clean and stringify all cell values to prevent mixed-type Arrow conversion errors
            # (e.g., float years like 2022.0 are formatted cleanly as "2022", and nulls become empty strings)
            for col in pd_blueprint.columns:
                pd_blueprint[col] = pd_blueprint[col].apply(
                    lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() 
                    else ("" if pd.isna(x) else str(x).strip())
                )
                
            df_blueprint = pl.from_pandas(pd_blueprint)
        except Exception as e:
            print(f"Failed to load Excel blueprint sheet '{sheet_name}'. Error: {e}")
            return pd.DataFrame()

        if os.path.exists(hai_path):
            try:
                df_hai = pl.read_csv(hai_path, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to load HAI file '{hai_path}'. Error: {e}")
                df_hai = pl.DataFrame()

        if os.path.exists(cpi_path):
            try:
                df_cpi = pl.from_pandas(pd.read_excel(cpi_path, sheet_name=0))
            except Exception as e:
                print(f"Warning: Failed to load CPI file '{cpi_path}'. Error: {e}")
                df_cpi = pl.DataFrame()

        if os.path.exists(mortgage_path):
            try:
                df_mortgage = pl.read_csv(mortgage_path, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to load Mortgage file '{mortgage_path}'. Error: {e}")
                df_mortgage = pl.DataFrame()

        df_acs_std = self.standardize_dataset(df_acs, "ACS")
        valid_geo_names = df_acs_std["NAME"].drop_nulls().unique().to_list()

        source_frames = [df_acs_std]

        if not df_comp.is_empty():
            df_comp_std = self.standardize_dataset(df_comp, "COMPONENTS")
            df_comp_std = df_comp_std.filter(pl.col("NAME").is_in(valid_geo_names))
            source_frames.append(df_comp_std)

        if not df_pyr.is_empty():
            df_pyr_std = self.standardize_dataset(df_pyr, "POP_PYRAMID")
            df_pyr_std = df_pyr_std.filter(pl.col("NAME").is_in(valid_geo_names))
            source_frames.append(df_pyr_std)

        if not df_hai.is_empty():
            df_hai_std = self.standardize_dataset(df_hai, "HAI")
            
            # Detect global reporting periods from HAI (Primary anchor for [Current Month] etc.)
            # We do this BEFORE geography filtering to get the full timeline availability of the dataset.
            # We filter for non-null Values to avoid picking up 'future' rows with only national lookup data.
            df_anchor_dates = df_hai_std.filter(pl.col("Value").is_not_null())
            available_months = df_anchor_dates.select(pl.col("MonthKey")).drop_nulls().unique().sort("MonthKey").to_series().to_list()
            
            if available_months:
                self.global_latest_month = available_months[-1]
                print(f"  [System] Global reporting period detected from HAI: {self.global_latest_month}")
                
                # Look for exactly 12 months ago
                target_prev = self.global_latest_month - 100
                if target_prev in available_months:
                    self.global_prev_month = target_prev
                else:
                    prev_cands = [m for m in available_months if m < self.global_latest_month]
                    self.global_prev_month = prev_cands[-1] if prev_cands else None
            
            # Now filter for requested geographies
            df_hai_std = df_hai_std.filter(pl.col("NAME").is_in(valid_geo_names))
            source_frames.append(df_hai_std)

        if not df_cpi.is_empty():
            source_frames.append(self.standardize_dataset(df_cpi, "CPI"))

        if not df_mortgage.is_empty():
            source_frames.append(self.standardize_dataset(df_mortgage, "MORTGAGE"))

        source_frames = [frame for frame in source_frames if not frame.is_empty()]
        if not source_frames:
            print("No valid standardized source frames available.")
            return pd.DataFrame()

        all_columns = []
        column_types = {}
        for frame in source_frames:
            for col, dtype in frame.schema.items():
                if col not in all_columns:
                    all_columns.append(col)
                    column_types[col] = dtype

        aligned_frames = []
        for frame in source_frames:
            # Ensure each frame uses the same dtype for every shared column.
            cast_columns = [pl.col(col).cast(column_types[col]).alias(col)
                            for col, dtype in frame.schema.items()
                            if col in column_types and dtype != column_types[col]]
            if cast_columns:
                frame = frame.with_columns(cast_columns)

            missing_columns = [col for col in all_columns if col not in frame.columns]
            if missing_columns:
                frame = frame.with_columns([
                    pl.lit(None, dtype=column_types[col]).alias(col)
                    for col in missing_columns
                ])
            aligned_frames.append(frame.select(all_columns))

        master_df = pl.concat(aligned_frames, how="vertical")
        
        final_results = []
        total_processing_time = 0
        metrics_processed = 0
        
        print(f"Blueprint loaded (sheet: '{sheet_name}'). Iterating through defined metrics...\n")

        for row in df_blueprint.iter_rows(named=True):
            # Check for 'Flag for Use' column and skip if False
            if "Flag for Use" in df_blueprint.columns:
                flag = row.get("Flag for Use")
                if flag is not None and str(flag).strip().lower() in ["false", "0", "f", "no"]:
                    print(f"Skipping metric '{row.get('Metric')}' due to Flag for Use = False")
                    continue
            
            math_debug_logs = []
            
            m_topic = row.get("Topic", "")
            m_name = row.get("Metric", "")
            m_type = row.get("Metric Type", "")
            m_desc = row.get("Description", "")
            d_source = row.get("Data Source", "")
            v_json = row.get("Variables", "")
            bp_comp = str(row.get("Comparison Period", "")) if row.get("Comparison Period") is not None else ""
            bp_curr = str(row.get("Current Period", "")) if row.get("Current Period") is not None else ""
            
            if not m_name: 
                continue
            
            start_time = time.time()
            
            geo_data, years_ctx, is_cat = self.calculate_metric_data(master_df, df_pyr, m_name, v_json, m_type, d_source, bp_curr=bp_curr, bp_comp=bp_comp)
            
            # Print debug info to trace geo_data contents and roles
            print(f"  [Debug] Metric: '{m_name}' | Available Roles in geo_data: {list(geo_data.keys()) if geo_data else 'None'}")
            
            if not geo_data or 'Focus' not in geo_data:
                # STRICT FALLBACK RULE: Only allow Broad fallback if the Focus geography is COMPLETELY absent in components dataset
                data_src_lower_check = str(d_source).lower() if d_source else ""
                is_comp_source = "component" in data_src_lower_check
                focus_absent_in_comp = df_comp_std.filter(pl.col("Role") == "Focus").is_empty() if not df_comp.is_empty() else True
                allow_broad_fallback = is_comp_source and focus_absent_in_comp
                if allow_broad_fallback and geo_data and 'Broad' in geo_data and len(geo_data['Broad']) > 0:
                    geo_data['Focus'] = geo_data['Broad'][0]
                    geo_data['Broad'] = []  # Omit broad summary to avoid redundancy
                    print(f"  [Fallback Applied] Metric '{m_name}' (components source): using Broad as Focus: {geo_data['Focus']['Name']}")
                else:
                    print(f"  [Skipped] Metric '{m_name}': Focus geography missing and data source ('{d_source}') does not permit Broad substitution.")
                    continue
                
            fn = geo_data['Focus']['Name']
            fv_val = geo_data['Focus'].get('Raw_Value')
            f_cat = geo_data['Focus'].get('Category')
            
            cy = years_ctx.get("latest", "Current Year")
            py = years_ctx.get("prev", "Previous Year")
            is_yoy = ("change" in m_name.lower() and "cumulative" not in m_name.lower())

            # --- DETERMINISTIC EXTRACTION (CYBORG) ---
            i_int = self.generate_internal_insight(fn, fv_val, m_name, m_type, is_yoy, cy, py, is_cat, f_cat)
            
            broad_geos = [g for g in geo_data.get('Broad', []) if not self.is_missing_value(g.get('Raw_Value'))]
            if broad_geos:
                i_brd = self.generate_comparative_insight(fn, broad_geos, fv_val, m_name, m_type, is_cat, f_cat)
            else: 
                i_brd = "N/A"
            
            bench_geos = geo_data.get('Benchmark', [])
            if bench_geos:
                i_bnc = self.generate_comparative_insight(fn, bench_geos, fv_val, m_name, m_type, is_cat, f_cat)
            else: 
                i_bnc = "N/A"
            
            peer_geos = geo_data.get('Peer', [])
            if peer_geos:
                i_per = self.generate_comparative_insight(fn, peer_geos, fv_val, m_name, m_type, is_cat, f_cat)
                
                peer_details = geo_data.get('Peer_Details', [])
                c_cat = peer_geos[0].get('Category') if peer_geos else None
                c_val = peer_geos[0].get('Raw_Value') if peer_geos else None
                
                is_pct_override = "%" in geo_data['Focus'].get('Formatted_Value', '')
                # --- DETERMINISTIC EXTRACTION FOR PEER DETAILED ---
                if is_cat:
                    i_per_det, m_per_det = self.generate_categorical_peer_detailed_insight(fv_val, f_cat, peer_details, c_cat, fn, m_name, m_type, is_percentage_override=is_pct_override)
                else:
                    is_pct = is_pct_override or "percent" in m_type.lower() or "rate" in m_type.lower() or "percent" in m_name.lower()
                    i_per_det, m_per_det = self.generate_peer_detailed_insight(fv_val, peer_details, c_val, fn, m_name, is_pct, m_type)
                    
                print(f"  [Math Debug] {m_per_det}")
                math_debug_logs.append(f"Peer Detailed: {m_per_det}")
            else: 
                i_per = "N/A"
                i_per_det = "N/A"
            
            # --- BUILD 3 MATH CONTEXT COLUMNS ---
            geo_data_raw = geo_data.get("raw", geo_data)
            cpi_debug_info = geo_data.get("cpi_debug_info", {})

            # 1. Final (adjusted as necessary, passed to the AI model)
            math_context_parts_final = [f"Focus value: {self.format_value(fv_val, m_name, m_type)}"]
            for g in broad_geos:
                if g.get('Raw_Value') is not None:
                    diff_v = fv_val - g['Raw_Value'] if fv_val is not None else None
                    if diff_v is not None:
                        diff_fmt = f"{abs(diff_v):.1f} pp" if ("percent" in m_type.lower() or "rate" in m_type.lower()) else self.format_value(abs(diff_v), m_name, m_type)
                        math_context_parts_final.append(f"{g['Name']}: {self.format_value(g['Raw_Value'], m_name, m_type)} (diff: {'+' if diff_v > 0 else '-'}{diff_fmt})")
            for g in bench_geos:
                if g.get('Raw_Value') is not None:
                    diff_v = fv_val - g['Raw_Value'] if fv_val is not None else None
                    if diff_v is not None:
                        diff_fmt = f"{abs(diff_v):.1f} pp" if ("percent" in m_type.lower() or "rate" in m_type.lower()) else self.format_value(abs(diff_v), m_name, m_type)
                        math_context_parts_final.append(f"{g['Name']}: {self.format_value(g['Raw_Value'], m_name, m_type)} (diff: {'+' if diff_v > 0 else '-'}{diff_fmt})")
            math_context = "; ".join(math_context_parts_final)
            math_debug_logs.append(f"Math Context: {math_context}")

            # 2. Raw (nominal/unadjusted)
            fv_val_raw = geo_data_raw.get('Focus', {}).get('Raw_Value', fv_val) if 'Focus' in geo_data_raw else fv_val
            broad_geos_raw = [g for g in geo_data_raw.get('Broad', []) if not self.is_missing_value(g.get('Raw_Value'))]
            bench_geos_raw = geo_data_raw.get('Benchmark', [])
            
            math_context_parts_raw = [f"Focus value: {self.format_value(fv_val_raw, m_name, m_type)}"]
            for g in broad_geos_raw:
                if g.get('Raw_Value') is not None:
                    diff_v = fv_val_raw - g['Raw_Value'] if fv_val_raw is not None else None
                    if diff_v is not None:
                        diff_fmt = f"{abs(diff_v):.1f} pp" if ("percent" in m_type.lower() or "rate" in m_type.lower()) else self.format_value(abs(diff_v), m_name, m_type)
                        math_context_parts_raw.append(f"{g['Name']}: {self.format_value(g['Raw_Value'], m_name, m_type)} (diff: {'+' if diff_v > 0 else '-'}{diff_fmt})")
            for g in bench_geos_raw:
                if g.get('Raw_Value') is not None:
                    diff_v = fv_val_raw - g['Raw_Value'] if fv_val_raw is not None else None
                    if diff_v is not None:
                        diff_fmt = f"{abs(diff_v):.1f} pp" if ("percent" in m_type.lower() or "rate" in m_type.lower()) else self.format_value(abs(diff_v), m_name, m_type)
                        math_context_parts_raw.append(f"{g['Name']}: {self.format_value(g['Raw_Value'], m_name, m_type)} (diff: {'+' if diff_v > 0 else '-'}{diff_fmt})")
            math_context_raw = "; ".join(math_context_parts_raw)

            # 3. Calculation Debug
            calc_parts = []
            if "current_period" in cpi_debug_info:
                calc_parts.append(f"Current Period: {cpi_debug_info['current_period']}")
            if "comparison_period" in cpi_debug_info and cpi_debug_info["comparison_period"] != "N/A":
                calc_parts.append(f"Comparison Period: {cpi_debug_info['comparison_period']}")
            if "current_cpi" in cpi_debug_info:
                calc_parts.append(f"Current CPI: {cpi_debug_info['current_cpi']}")
            if "comparison_cpi" in cpi_debug_info and cpi_debug_info["comparison_cpi"] != "N/A":
                calc_parts.append(f"Comparison CPI: {cpi_debug_info['comparison_cpi']}")
            if "calculation" in cpi_debug_info:
                calc_parts.append(f"Calculation: {cpi_debug_info['calculation']}")
            math_context_calculation = "; ".join(calc_parts) if calc_parts else "Calculation: Nominal (no CPI adjustment)"

            # --- LLM SYNTHESIS FOR OVERALL INSIGHT ---
            synth_prompt = f"""You are an executive data analyst. Synthesize the facts below about '{m_name}' into ONE professional, fluid summary sentence.
            
            FOCUS GEOGRAPHY: {fn}
            
            FACTS (use these as your source of truth):
            1. Internal: {i_int} 
            2. Broad: {i_brd} 
            3. Benchmarks: {i_bnc} 
            4. Peers: {i_per_det}
            
            KEY FIGURES (reference these for magnitudes where relevant):
            {math_context}
            
            RULES:
            1. The summary MUST be about "{fn}" as the primary subject. The broad geography and benchmarks are context only — never write a sentence that makes the broad geography (e.g. the county or MSA) the main subject.
            2. Start directly with "{fn}" — not with the broad geography or any other place.
            3. Include specific numbers or magnitudes where they add meaningful context (e.g. "by 2.1 percentage points", "at 11,077"). Do NOT list every single figure — only the most impactful ones.
            4. DO NOT compare a geography to itself.
            5. Synthesize available comparisons elegantly into one flowing sentence or two short sentences at most.
            6. If any comparison (Broad, Benchmarks, or Peers) is 'N/A', simply ignore that category.
            7. Use Title Case for all geography names (e.g. "Pierce County", "Seattle MSA", "United States").
            8. Use lowercase for demographic/category terms (e.g. "female", "natural change", "45-54 age group").
            9. AVOID storytelling, conversational, or narrative framing/introductory phrases (e.g., "demographic analysis reveals", "according to the data", "interestingly", "notably", "the numbers show that", "a closer look shows that"). State the analytical facts directly and declaratively.
            10. The output must be crisp, professional, objective, and directly insertable into an executive-level dashboard card.
            
            Output STRICTLY as a valid JSON object: {{ "overall_insight": "your sentence here" }}
            """
            
            i_over_raw = self.generate_text(synth_prompt, as_json=True, json_key="overall_insight", max_words=150) if i_int != "N/A" else "N/A"

            all_peer_names = [p["Name"] for p in geo_data.get('Peer_Details', [])]
            broad_names = [g["Name"] for g in broad_geos] if broad_geos else []
            bench_names = [g["Name"] for g in bench_geos] if bench_geos else []
            peer_names = [g["Name"] for g in peer_geos] if peer_geos else []
            valid_geos_over = list({g for g in [fn] + broad_names + bench_names + peer_names + all_peer_names if g})
            i_over = self.apply_capitalization_layer(i_over_raw, valid_geos_over)
            
            if cy:
                bp_comp = bp_comp.replace("[Current Year]", cy)
                bp_curr = bp_curr.replace("[Current Year]", cy)
            if py:
                bp_comp = bp_comp.replace("[Previous Year]", py)
                bp_curr = bp_curr.replace("[Previous Year]", py)

            bp_comp = self.format_period_placeholders(bp_comp)
            bp_curr = self.format_period_placeholders(bp_curr)

            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            total_processing_time += processing_time
            metrics_processed += 1
            
            print(f"\n[✓] Processed Metric {metrics_processed}: '{m_name}' in {processing_time}s")
            print(f"  [Internal]   {i_int}")
            print(f"  [Broad]      {i_brd}")
            print(f"  [Benchmarks] {i_bnc}")
            print(f"  [Peers]      {i_per}")
            print(f"  [Peers Det]  {i_per_det}")
            print(f"  [Overall]    {i_over}")
            
            final_results.append({
                "Topic": m_topic,
                "Comparison Period": bp_comp, 
                "Current Period": bp_curr,    
                "Data Source": d_source,
                "Variables": v_json,
                "Metric": m_name,
                "Metric Type": m_type,
                "Description": m_desc,
                "Internal Insight": i_int,
                "Comparative Insight (Broad)": i_brd,
                "Comparative Insight (Benchmarks)": i_bnc,
                "Comparative Insight (Peers)": i_per,
                "Comparative Insight (Peers - Detailed)": i_per_det,
                "Overall Insight": i_over,
                "Math Debug - Final": math_context,
                "Math Debug - Raw": math_context_raw,
                "Math Debug - Calculation": math_context_calculation
            })

        if metrics_processed == 0: 
            return pd.DataFrame()

        df_final = pd.DataFrame(final_results)

        # Apply month placeholder formatting to insight columns
        insight_columns = [
            "Internal Insight", "Comparative Insight (Broad)", 
            "Comparative Insight (Benchmarks)", "Comparative Insight (Peers)", 
            "Comparative Insight (Peers - Detailed)", "Overall Insight"
        ]
        for col in insight_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].apply(self.format_period_placeholders)

        # --- TOPIC SUMMARIES ---
        print("\nSynthesizing Topic Summaries...")
        topic_summaries = {}
        for topic, group in df_final.groupby("Topic"):
            if not topic: continue
            all_overall_insights = group["Overall Insight"].dropna().tolist()
            insights_str = "\n".join([f"- {insight}" for insight in all_overall_insights if insight != "N/A"])
            
            if not insights_str: 
                topic_summaries[topic] = "N/A"
                continue

            # Also pass the per-metric math debug for numerical grounding in the topic prompt
            topic_math = "\n".join([
                f"- {row2['Metric']}: {row2['Math Debug - Final']}"
                for _, row2 in group.iterrows()
                if row2.get('Math Debug - Final') and row2['Math Debug - Final'] != "Deterministic Extraction (Cyborg Arch)"
            ])
            topic_prompt = f"""You are an executive data analyst writing a single cohesive summary paragraph for the topic: {topic}.
Synthesize the following insights into a smooth, professional paragraph.

INSIGHTS:
{insights_str}

KEY FIGURES (use the most impactful numbers to ground the paragraph -- do not repeat every figure):
{topic_math if topic_math else 'No additional figures available.'}

RULES:
1. Keep the paragraph concise and use no more than 150 words.
2. Include specific numbers where they meaningfully support the narrative (e.g. magnitudes, differences).
3. Avoid bullet points. Transitions between sentences must feel natural.
4. Use Title Case for geography names. Use lowercase for demographic/category labels.
5. Do NOT mention data availability or missing comparisons.
6. Do not invent new geography names or project-specific place names beyond the provided list.
7. AVOID storytelling, conversational, or narrative framing/introductory phrases (e.g., "demographic analysis reveals", "according to the data", "interestingly", "notably", "the data shows that", "it is important to note that"). State the analytical facts directly and declaratively.
8. The output must be polished, objective, and directly insertable into an executive-level dashboard topic overview card.

Output STRICTLY as a valid JSON object formatted as: {{ "topic_summary": "your paragraph here" }}"""
            
            raw_summary = self.generate_text(topic_prompt, as_json=True, json_key="topic_summary")
            polished_summary = self.apply_capitalization_layer(raw_summary, [])
            topic_summaries[topic] = polished_summary
            print(f"  [{topic}] Summary generated.\n")

        df_final["Topic Summary"] = df_final["Topic"].map(topic_summaries).fillna("N/A")

        # --- COMPLETE SUMMARY ---
        print("\nSynthesizing Complete Executive Summary...")
        all_topics_combined = "\n".join([f"{t}: {s}" for t, s in topic_summaries.items() if s != "N/A"])
        
        complete_summary_prompt = f"""You are an executive data analyst writing a single, high-level executive summary for a dashboard.
Synthesize the following topic-level summaries into ONE cohesive paragraph that highlights the most critical insights.

Topic Summaries:
{all_topics_combined}

RULES:
1. Keep the executive summary concise and use no more than 150 words.
2. Include the most impactful specific figures where they strengthen the narrative. Do not enumerate every number.
3. Do not use bullet points. Keep it professional, objective, and insightful.
4. Use Title Case for geography names. Use lowercase for demographic/category labels.
5. Do not invent or add project-specific place names beyond the provided list.
6. AVOID storytelling, conversational, or narrative framing/introductory phrases (e.g., "demographic analysis reveals", "according to the data", "interestingly", "notably", "the data shows that", "it is important to note that"). State the analytical facts directly and declaratively.
7. The output must be polished, objective, and directly insertable into an executive-level dashboard home-screen summary widget.

Output STRICTLY as a valid JSON object formatted as: {{ "complete_summary": "your executive summary here" }}"""

        complete_raw = self.generate_text(complete_summary_prompt, as_json=True, json_key="complete_summary", max_words=150)
        complete_polished = self.apply_grammar_layer(complete_raw)
        
        print(f"  [Complete Summary] Summary generated.\n")
        df_final["Complete Summary"] = complete_polished

        columns_ordered = [
            "Topic", "Topic Summary", "Complete Summary", "Comparison Period", "Current Period", 
            "Data Source", "Variables", "Metric", "Metric Type", "Description", 
            "Internal Insight", "Comparative Insight (Broad)", 
            "Comparative Insight (Benchmarks)", "Comparative Insight (Peers)", 
            "Comparative Insight (Peers - Detailed)", "Overall Insight", 
            "Math Debug - Final", "Math Debug - Raw", "Math Debug - Calculation"
        ]
        df_final = df_final[[col for col in columns_ordered if col in df_final.columns]]
        
        avg_time = round(total_processing_time / metrics_processed, 2)
        print("\n" + "="*50)
        print(f"PIPELINE COMPLETE: {metrics_processed} Metrics processed.")
        print(f"Average Processing Time: {avg_time} seconds/metric (Cyborg Mode)")
        print("="*50 + "\n")
        
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        return df_final


def run_pipeline(blueprint_path: str = 'Metric Topics (DRAFT).xlsx',
                 acs_path: str = 'examples/ACS_Series_Polars.csv',
                 components_path: str = 'examples/components_of_change (4).csv',
                 pyramid_path: str = 'examples/population_pyramid.csv',
                 sheet_name: str = 'AI Summary',
                 mode: str = 'gemini',
                 model_name: str = 'gemma3',
                 gemini_model: str = 'gemini-2.5-flash',
                 api_key: str = None,
                 vertexai: bool = None,
                 project: str = None,
                 location: str = None,
                 hai_path: str = 'examples/HAI.csv',
                 cpi_path: str = 'examples/cpi.xlsx',
                 mortgage_path: str = 'examples/MortgageRates.csv',
                 output_path: str = 'AI_Summary.csv') -> pd.DataFrame:
    """
    Module-level convenience function to run the pipeline.
    Useful for importing and executing in other scripts.
    """
    pipeline = AISummaryPipeline(
        mode=mode, 
        model_name=model_name, 
        gemini_model=gemini_model, 
        api_key=api_key,
        vertexai=vertexai,
        project=project,
        location=location
    )
    return pipeline.run(
        blueprint_path=blueprint_path,
        acs_path=acs_path,
        components_path=components_path,
        pyramid_path=pyramid_path,
        hai_path=hai_path,
        cpi_path=cpi_path,
        mortgage_path=mortgage_path,
        sheet_name=sheet_name,
        output_path=output_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the AI Summary Generation Pipeline.")
    parser.add_argument("--blueprint", default="Metric Topics (DRAFT).xlsx", help="Path to Excel blueprint file")
    parser.add_argument("--acs", default="examples/ACS_Series_Polars.csv", help="Path to ACS series CSV file")
    parser.add_argument("--components", default="examples/components_of_change (4).csv", help="Path to components of change CSV file")
    parser.add_argument("--pyramid", default="examples/population_pyramid.csv", help="Path to population pyramid CSV file")
    parser.add_argument("--hai", default="examples/HAI.csv", help="Path to HAI monthly CSV file")
    parser.add_argument("--cpi", default="examples/cpi.xlsx", help="Path to CPI Excel workbook")
    parser.add_argument("--mortgage", default="examples/MortgageRates.csv", help="Path to Mortgage Rates CSV file")
    parser.add_argument("--sheet", default="AI Summary", help="Excel sheet name to use (default: AI Summary)")
    parser.add_argument("--mode", default="gemini", choices=["gemini", "ollama"], help="Model mode (gemini or ollama)")
    parser.add_argument("--model-name", default="gemma3", help="Ollama model name (default: gemma3)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key (falls back to GEMINI_API_KEY env var)")
    parser.add_argument("--vertexai", action="store_true", default=None, help="Force Vertex AI mode (instead of Developer API)")
    parser.add_argument("--no-vertexai", action="store_false", dest="vertexai", help="Force disable Vertex AI mode")
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud project ID for Vertex AI")
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION"), help="Google Cloud location for Vertex AI")
    parser.add_argument("--output", default="AI_Summary.csv", help="Path to save output CSV file")
    
    args = parser.parse_args()
    
    run_pipeline(
        blueprint_path=args.blueprint,
        acs_path=args.acs,
        components_path=args.components,
        pyramid_path=args.pyramid,
        hai_path=args.hai,
        cpi_path=args.cpi,
        mortgage_path=args.mortgage,
        sheet_name=args.sheet,
        mode=args.mode,
        model_name=args.model_name,
        gemini_model=args.gemini_model,
        api_key=args.api_key,
        vertexai=args.vertexai,
        project=args.project,
        location=args.location,
        output_path=args.output
    )
