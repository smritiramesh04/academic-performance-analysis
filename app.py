from pathlib import Path
import re
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EduPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* MAIN APPLICATION */

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}


/* STREAMLIT HEADER */

header[data-testid="stHeader"] {
    background-color: #0B1120;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1F2937;
}

section[data-testid="stSidebar"] * {
    color: #D1D5DB;
}


/* HEADINGS */

h1, h2, h3 {
    color: #F9FAFB !important;
}

p {
    color: #CBD5E1;
}


/* BRAND */

.brand-name {
    color: #38BDF8;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}


/* MAIN TITLE */

.dashboard-title {
    background: linear-gradient(135deg, #38BDF8 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
    line-height: 1.2;
}


/* SUBTITLE */

.dashboard-subtitle {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 35px;
}


/* KPI METRIC CARDS */

.kpi-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}

.kpi-card.blue::before { background: linear-gradient(90deg, #38BDF8, #0EA5E9); }
.kpi-card.purple::before { background: linear-gradient(90deg, #A78BFA, #8B5CF6); }
.kpi-card.green::before { background: linear-gradient(90deg, #34D399, #10B981); }
.kpi-card.red::before { background: linear-gradient(90deg, #F87171, #EF4444); }
.kpi-card.amber::before { background: linear-gradient(90deg, #FBBF24, #F59E0B); }
.kpi-card.pink::before { background: linear-gradient(90deg, #F472B6, #EC4899); }

.kpi-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}

.kpi-label {
    font-size: 13px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.kpi-delta {
    font-size: 12px;
    margin-top: 6px;
    font-weight: 600;
}

.kpi-delta.up { color: #34D399; }
.kpi-delta.down { color: #F87171; }
.kpi-delta.neutral { color: #94A3B8; }


/* SECTION HEADERS */

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}

.section-header .icon {
    font-size: 22px;
}

.section-header .text {
    font-size: 20px;
    font-weight: 700;
    color: #F1F5F9;
}


/* AT-RISK BADGE */

.risk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.risk-badge.high {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.risk-badge.medium {
    background: rgba(251, 191, 36, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.risk-badge.low {
    background: rgba(52, 211, 153, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}


/* SIDEBAR RADIO */

div[role="radiogroup"] label {
    padding-top: 7px;
    padding-bottom: 7px;
}

div[role="radiogroup"] label p {
    color: #CBD5E1 !important;
    font-size: 15px;
}


/* DIVIDER */

hr {
    border-color: #1F2937 !important;
}


/* REMOVE STREAMLIT FOOTER */

footer {
    visibility: hidden;
}


/* PLOTLY CHART CONTAINER */

.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}


/* DOWNLOAD BUTTON */

.stDownloadButton > button {
    background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
    color: #E2E8F0;
    border: 1px solid #475569;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #334155 0%, #475569 100%);
    border-color: #38BDF8;
    color: #F8FAFC;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER: PLOTLY DARK THEME LAYOUT
# --------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#CBD5E1", size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#1E293B",
        tickfont=dict(color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#1E293B",
        tickfont=dict(color="#94A3B8"),
    ),
    hoverlabel=dict(
        bgcolor="#1E293B",
        font_size=13,
        font_color="#F1F5F9",
        bordercolor="#334155",
    ),
    legend=dict(font=dict(color="#94A3B8")),
    coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")),
)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply the dark theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# --------------------------------------------------
# HELPER: KPI CARD HTML
# --------------------------------------------------

def kpi_card(
    icon: str,
    value: str,
    label: str,
    color: str = "blue",
    delta: str | None = None,
    delta_direction: str = "neutral",
) -> str:
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_direction}">{delta}</div>'
    return (
        f'<div class="kpi-card {color}">'
        f'  <div class="kpi-icon">{icon}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-label">{label}</div>'
        f'  {delta_html}'
        f'</div>'
    )


def section_header(icon: str, text: str) -> str:
    return (
        f'<div class="section-header">'
        f'  <span class="icon">{icon}</span>'
        f'  <span class="text">{text}</span>'
        f'</div>'
    )


# --------------------------------------------------
# HELPER: CSV DOWNLOAD
# --------------------------------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes for download."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


# --------------------------------------------------
# DATA LOADING (CACHED)
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame with caching."""
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_sample_csv(path: str) -> pd.DataFrame:
    """Load the bundled sample CSV with caching."""
    return pd.read_csv(path)


# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------

st.sidebar.markdown("## EduPulse")
st.sidebar.caption("Academic Intelligence Platform")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload class or school data",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = load_csv(uploaded_file.getvalue(), uploaded_file.name)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.sidebar.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
    st.sidebar.success(f"Loaded {uploaded_file.name}")
else:
    sample_path = Path(__file__).resolve().parent / "student_data.csv"
    if not sample_path.is_file():
        st.sidebar.error(f"Sample dataset not found: {sample_path}")
        st.stop()
    data = load_sample_csv(str(sample_path))
    st.sidebar.info("Using sample dataset")

st.sidebar.divider()

if data.empty or len(data.columns) == 0:
    st.error("This CSV has no rows or columns to analyze.")
    st.stop()

data.columns = [str(column).strip() for column in data.columns]


def normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


numeric_values = {column: numeric_series(data[column]) for column in data.columns}
numeric_columns = [
    column
    for column, values in numeric_values.items()
    if values.notna().sum() > 0
    and values.notna().mean() >= 0.8
]
dataset_signature = (
    uploaded_file.name if uploaded_file else "sample",
    tuple(data.columns),
    len(data),
    tuple(
        (column, float(numeric_values[column].min()), float(numeric_values[column].max()))
        for column in numeric_columns
    ),
)
widget_namespace = str(abs(hash(dataset_signature)))

score_keywords = (
    "score", "mark", "grade", "exam", "test", "quiz", "assessment", "result"
)
non_score_keywords = (
    "attendance", "study_hour", "participation", "age", "year", "semester",
    "student_id", "roll", "index", "class_id",
)
suggested_scores = [
    column
    for column in numeric_columns
    if any(word in normalize_column_name(column) for word in score_keywords)
]
if not suggested_scores:
    suggested_scores = [
        column
        for column in numeric_columns
        if not any(word in normalize_column_name(column) for word in non_score_keywords)
    ]

student_candidates = [
    column
    for column in data.columns
    if any(
        word in normalize_column_name(column)
        for word in ("student_name", "student_id", "student", "learner", "name", "roll")
    )
]
subject_candidates = [
    column
    for column in data.columns
    if any(
        word in normalize_column_name(column)
        for word in ("subject", "course", "paper", "module", "discipline")
    )
]

# --- Smart auto-detection (zero config for most datasets) ---
student_column = student_candidates[0] if student_candidates else None
subject_column = subject_candidates[0] if subject_candidates else None
selected_score_columns = [
    c for c in suggested_scores if c not in (student_column, subject_column)
]

# --- Page navigation (primary sidebar control) ---
page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Student Explorer",
        "Compare Students",
        "Student Support",
        "Academic Insights",
    ],
)

st.sidebar.divider()

# --- Advanced settings (collapsed for power users) ---
with st.sidebar.expander("Advanced settings"):
    student_options = ["Use row number"] + data.columns.tolist()
    student_default = (
        student_options.index(student_candidates[0])
        if student_candidates
        else 0
    )
    student_column = st.selectbox(
        "Student name / ID column",
        student_options,
        index=student_default,
        key=f"student_column_{widget_namespace}",
    )
    student_column = None if student_column == "Use row number" else student_column

    subject_options = ["No subject column"] + data.columns.tolist()
    subject_default = (
        subject_options.index(subject_candidates[0]) if subject_candidates else 0
    )
    subject_column = st.selectbox(
        "Subject column",
        subject_options,
        index=subject_default,
        key=f"subject_column_{widget_namespace}",
    )
    subject_column = None if subject_column == "No subject column" else subject_column

    selected_score_columns = st.multiselect(
        "Score columns",
        [
            column
            for column in numeric_columns
            if column not in (student_column, subject_column)
        ],
        default=[
            column
            for column in suggested_scores
            if column not in (student_column, subject_column)
        ],
        key=f"score_columns_{widget_namespace}",
    )

if not selected_score_columns:
    st.warning(
        "No score fields detected. Open **Advanced settings** in the "
        "sidebar to choose score columns."
    )
    st.stop()

score_records = pd.DataFrame(
    {
        "_student": (
            data[student_column].astype("string").str.strip()
            if student_column
            else pd.Series(
                [f"Student {number}" for number in range(1, len(data) + 1)],
                index=data.index,
                dtype="string",
            )
        )
    }
)
missing_student_ids = (
    score_records["_student"].isna() | score_records["_student"].eq("")
)
score_records.loc[missing_student_ids, "_student"] = [
    f"Unnamed student {number}" for number in range(1, int(missing_student_ids.sum()) + 1)
]
score_records["_row"] = range(1, len(data) + 1)
for column in selected_score_columns:
    score_records[column] = numeric_values[column].to_numpy()

long_scores = score_records.melt(
    id_vars=["_student", "_row"],
    value_vars=selected_score_columns,
    var_name="_score_field",
    value_name="Score",
)
long_scores["Score"] = pd.to_numeric(
    pd.concat(
        [
            numeric_values[column].reset_index(drop=True)
            for column in selected_score_columns
        ],
        ignore_index=True,
    ),
    errors="coerce",
).to_numpy()
if subject_column and len(selected_score_columns) == 1:
    long_scores["Subject"] = data[subject_column].astype("string").fillna("Unspecified")
else:
    long_scores["Subject"] = long_scores["_score_field"].astype("string")

long_scores = long_scores.dropna(subset=["Score"])
if long_scores.empty:
    st.warning("The selected score columns do not contain any readable numeric values.")
    st.stop()

observed_max = float(long_scores["Score"].max())
common_scales = [
    5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 80.0, 100.0,
    150.0, 200.0, 500.0, 1000.0,
]
default_scale = next(
    (scale for scale in common_scales if scale >= observed_max),
    observed_max,
)

# Sensible defaults (work without touching anything)
score_scale = default_scale
support_cutoff = float(score_scale * 0.50)
high_cutoff = float(score_scale * 0.75)
threshold_step = max(float(score_scale) / 100, 0.1)
threshold_namespace = f"{widget_namespace}_{score_scale:g}"

subject_values = sorted(long_scores["Subject"].dropna().astype(str).unique())
selected_subjects = subject_values

# Power-user overrides (collapsed)
with st.sidebar.expander("Thresholds & filters"):
    score_scale = st.number_input(
        "Max score on this scale",
        min_value=1.0,
        max_value=10000.0,
        value=default_scale,
        step=1.0,
        key=f"score_scale_{widget_namespace}",
    )
    threshold_step = max(float(score_scale) / 100, 0.1)
    support_cutoff = st.slider(
        "Needs support below",
        min_value=0.0,
        max_value=float(score_scale) - threshold_step,
        value=float(score_scale * 0.5),
        step=threshold_step,
        key=f"support_cutoff_{threshold_namespace}",
    )
    high_cutoff = st.slider(
        "Strong performance from",
        min_value=float(support_cutoff),
        max_value=float(score_scale),
        value=max(float(support_cutoff), float(score_scale * 0.75)),
        step=threshold_step,
        key=f"high_cutoff_{threshold_namespace}_{support_cutoff:g}",
    )
    selected_subjects = st.multiselect(
        "Filter subjects / score fields",
        subject_values,
        default=subject_values,
        key=(
            f"subjects_{widget_namespace}_"
            f"{abs(hash((tuple(selected_score_columns), tuple(subject_values))))}"
        ),
    )

st.sidebar.caption("Scores only. Study hours are not used for support bands.")

subject_scores = long_scores[
    long_scores["Subject"].astype(str).isin(selected_subjects)
].copy()
if subject_scores.empty:
    st.info("Select at least one subject in the sidebar to view student results.")
    st.stop()

student_subject_scores = (
    subject_scores.pivot_table(
        index="_student",
        columns="Subject",
        values="Score",
        aggfunc="mean",
        observed=False,
    )
    .reset_index()
)
student_summary = (
    subject_scores.groupby("_student", as_index=False)
    .agg(
        Overall=("Score", "mean"),
        Std_Dev=("Score", "std"),
        Assessments=("Score", "count"),
    )
)
student_summary["Std_Dev"] = student_summary["Std_Dev"].fillna(0.0)


def performance_band(score: float) -> str:
    if score < support_cutoff:
        return "Needs support"
    if score < high_cutoff:
        return "On track"
    return "Strong"


student_summary["Performance"] = student_summary["Overall"].map(performance_band)

# Percentile ranking
student_summary["Percentile"] = (
    student_summary["Overall"].rank(pct=True).mul(100).round(1)
)

# At-risk early warning: within 10 % of scale above the support cutoff
at_risk_margin = score_scale * 0.10
student_summary["At_Risk"] = student_summary["Overall"].apply(
    lambda s: (
        "High" if s < support_cutoff
        else "Medium" if s < support_cutoff + at_risk_margin
        else "Low"
    )
)

student_summary = student_summary.merge(student_subject_scores, on="_student", how="left")

filtered_scores = subject_scores[
    subject_scores["_student"].isin(student_summary["_student"])
].copy()


def student_advice(student_scores_series: pd.Series, overall: float) -> str:
    available = student_scores_series.dropna().sort_values()
    weakest = available.index[0] if not available.empty else None
    strongest = available.index[-1] if not available.empty else None

    if overall >= high_cutoff:
        strength = (
            f" Your strongest result is {strongest} ({available.iloc[-1]:g})."
            if strongest is not None
            else ""
        )
        stretch = (
            f" For your next challenge, build on {weakest} "
            f"({available.iloc[0]:g}) with extension questions."
            if weakest is not None and weakest != strongest
            else " Try an extension question or explain a topic to a classmate."
        )
        return (
            "Strong work. Keep using the learning approaches that are working."
            + strength
            + stretch
        )
    if overall >= support_cutoff:
        strength = (
            f" Use your stronger result in {strongest} "
            f"({available.iloc[-1]:g}) as a model."
            if strongest is not None and strongest != weakest
            else ""
        )
        focus = (
            f" Your next focus is {weakest} ({available.iloc[0]:g}): "
            "practise a few questions on one skill and review any mistakes."
            if weakest
            else " Review missed questions and practise one skill at a time."
        )
        return (
            "You are building a solid foundation. Keep reviewing missed "
            "questions and check your progress after the next assessment."
            + strength
            + focus
        )

    strength = (
        f" Your strongest current result is {strongest} "
        f"({available.iloc[-1]:g}); use that confidence as you learn."
        if strongest is not None and strongest != weakest
        else ""
    )
    focus = (
        f" Start with {weakest} ({available.iloc[0]:g}) and revisit its "
        "prerequisite skills."
        if weakest
        else " Ask your teacher to help identify the first skill to revisit."
    )
    return (
        "Your next step is focused support, not trying to fix everything at "
        "once. Meet with your teacher to identify one gap, learn that skill "
        "from a worked example, practise guided questions, then try a similar "
        "question independently. Review mistakes together and check this skill "
        "again after the next assessment."
        + strength
        + focus
    )


student_advice_by_id = {
    row["_student"]: student_advice(
        student_subject_scores.loc[
            student_subject_scores["_student"] == row["_student"]
        ].iloc[0].drop(labels="_student"),
        float(row["Overall"]),
    )
    for _, row in student_summary.iterrows()
}
student_summary["Personal advice"] = student_summary["_student"].map(
    student_advice_by_id
)




# --------------------------------------------------
# DISPLAY HELPERS
# --------------------------------------------------

def display_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable for display (drop internal ones)."""
    hide = {"_student", "_row", "_score_field"}
    return [c for c in df.columns if c not in hide]


def display_df(df: pd.DataFrame, rename_student: bool = True) -> pd.DataFrame:
    """Prepare a DataFrame for display."""
    out = df.copy()
    if rename_student and "_student" in out.columns:
        out = out.rename(columns={"_student": "Student"})
    cols = display_columns(out)
    return out[cols]


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="dashboard-title">Performance Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-subtitle">
        Recognize strengths, find where students need support, and make
        data-informed plans for academic growth.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DASHBOARD PAGES
# --------------------------------------------------

if page == "Overview":

    st.markdown(section_header("\U0001f310", "Overview"), unsafe_allow_html=True)
    st.caption(
        "A class-wide view of achievement, score-field strengths, and learning needs."
    )

    if filtered_scores.empty:
        st.info(
            "No results match the selected subjects and score range. "
            "Adjust the sidebar filters."
        )
    else:
        # --- KPI METRICS ---
        counts = student_summary["Performance"].value_counts().reindex(
            ["Strong", "On track", "Needs support"], fill_value=0
        )
        total_students = int(student_summary["_student"].nunique())
        avg_score = float(student_summary["Overall"].mean())
        median_score = float(student_summary["Overall"].median())
        strong_count = int(counts["Strong"])
        on_track_count = int(counts["On track"])
        needs_support_count = int(counts["Needs support"])
        at_risk_count = int((student_summary["At_Risk"] == "Medium").sum())

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.markdown(
            kpi_card("\U0001f393", f"{total_students:,}", "Students", "blue"),
            unsafe_allow_html=True,
        )
        k2.markdown(
            kpi_card(
                "\U0001f4ca", f"{avg_score:.1f}", "Average", "purple",
                f"Median {median_score:.1f}", "neutral",
            ),
            unsafe_allow_html=True,
        )
        k3.markdown(
            kpi_card(
                "⭐", f"{strong_count:,}", "Strong", "green",
                f"{strong_count / max(total_students, 1) * 100:.0f}%", "up",
            ),
            unsafe_allow_html=True,
        )
        k4.markdown(
            kpi_card(
                "✅", f"{on_track_count:,}", "On Track", "blue",
                f"{on_track_count / max(total_students, 1) * 100:.0f}%", "neutral",
            ),
            unsafe_allow_html=True,
        )
        k5.markdown(
            kpi_card(
                "⚠️", f"{needs_support_count:,}", "Need Support", "red",
                f"{needs_support_count / max(total_students, 1) * 100:.0f}%", "down",
            ),
            unsafe_allow_html=True,
        )
        k6.markdown(
            kpi_card(
                "\U0001f6a8", f"{at_risk_count:,}", "At-Risk", "amber",
                "Near cutoff", "down" if at_risk_count > 0 else "neutral",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- CHARTS ---
        left, right = st.columns(2)
        with left:
            st.markdown(
                section_header("\U0001f4c8", "Class Performance Breakdown"),
                unsafe_allow_html=True,
            )
            band_df = counts.reset_index()
            band_df.columns = ["Performance", "Students"]
            color_map = {
                "Strong": "#34D399",
                "On track": "#38BDF8",
                "Needs support": "#F87171",
            }
            fig_band = px.bar(
                band_df,
                x="Performance",
                y="Students",
                color="Performance",
                color_discrete_map=color_map,
                text="Students",
            )
            fig_band.update_traces(textposition="outside", textfont_size=14)
            fig_band.update_layout(showlegend=False)
            apply_plotly_theme(fig_band)
            st.plotly_chart(fig_band, use_container_width=True)

        with right:
            st.markdown(
                section_header("\U0001f4da", "Average by Subject / Score Field"),
                unsafe_allow_html=True,
            )
            subject_means = (
                filtered_scores.groupby("Subject", observed=False)["Score"]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            subject_means.columns = ["Subject", "Average"]
            fig_subj = px.bar(
                subject_means,
                x="Average",
                y="Subject",
                orientation="h",
                text="Average",
                color="Average",
                color_continuous_scale=["#F87171", "#FBBF24", "#34D399"],
            )
            fig_subj.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_subj.update_layout(coloraxis_showscale=False)
            apply_plotly_theme(fig_subj)
            st.plotly_chart(fig_subj, use_container_width=True)

        # --- SCORE DISTRIBUTION ---
        st.markdown(
            section_header("\U0001f4c0", "Score Distribution"),
            unsafe_allow_html=True,
        )
        fig_hist = px.histogram(
            filtered_scores,
            x="Score",
            nbins=min(20, max(5, int(filtered_scores["Score"].nunique()))),
            color_discrete_sequence=["#A78BFA"],
            marginal="box",
        )
        fig_hist.update_layout(bargap=0.05)
        fig_hist.add_vline(
            x=support_cutoff, line_dash="dash", line_color="#F87171",
            annotation_text="Support cutoff", annotation_position="top left",
            annotation_font_color="#F87171",
        )
        fig_hist.add_vline(
            x=high_cutoff, line_dash="dash", line_color="#34D399",
            annotation_text="Strong cutoff", annotation_position="top right",
            annotation_font_color="#34D399",
        )
        apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- STUDENT TABLE ---
        st.markdown(
            section_header("\U0001f4dd", "Student Overview"),
            unsafe_allow_html=True,
        )
        overview_display = display_df(student_summary)
        st.dataframe(overview_display, use_container_width=True, hide_index=True)

        col_dl1, col_dl2, _ = st.columns([1, 1, 4])
        with col_dl1:
            st.download_button(
                "Download student report",
                to_csv_bytes(overview_display),
                file_name="edupulse_student_overview.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                "Download filtered scores",
                to_csv_bytes(
                    filtered_scores[["_student", "Subject", "Score"]].rename(
                        columns={"_student": "Student"}
                    )
                ),
                file_name="edupulse_filtered_scores.csv",
                mime="text/csv",
            )

        with st.expander("Uploaded rows and detected fields"):
            st.write("Columns:", data.columns.tolist())
            st.dataframe(data, use_container_width=True, hide_index=True)


elif page == "Student Explorer":

    st.markdown(
        section_header("\U0001f50d", "Student Explorer"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Search, filter, and explore individual student performance across all "
        "score fields."
    )

    if student_summary.empty:
        st.info("No students match the current filters.")
    else:
        # --- SEARCH BAR ---
        search_query = st.text_input(
            "Search students",
            placeholder="Type a name or ID to filter…",
            key="student_search",
        )
        explorer_df = student_summary.copy()
        if search_query.strip():
            explorer_df = explorer_df[
                explorer_df["_student"]
                .str.lower()
                .str.contains(search_query.strip().lower(), na=False)
            ]

        if explorer_df.empty:
            st.info(f'No students match "{search_query}".')
        else:
            st.dataframe(
                display_df(explorer_df),
                use_container_width=True,
                hide_index=True,
            )

            selected_student = st.selectbox(
                "View an individual student",
                explorer_df["_student"].tolist(),
            )
            selected_row = explorer_df.loc[
                explorer_df["_student"] == selected_student
            ].iloc[0]

            st.markdown(f"#### {selected_student}")

            # Styled metric cards
            m1, m2, m3, m4 = st.columns(4)
            band = str(selected_row["Performance"])
            band_color = {
                "Strong": "green",
                "On track": "blue",
                "Needs support": "red",
            }.get(band, "blue")
            m1.markdown(
                kpi_card(
                    "\U0001f3af", f"{selected_row['Overall']:.1f}",
                    "Overall Score", band_color,
                ),
                unsafe_allow_html=True,
            )
            m2.markdown(
                kpi_card("\U0001f3c5", band, "Performance", band_color),
                unsafe_allow_html=True,
            )
            m3.markdown(
                kpi_card(
                    "\U0001f4c8", f"P{selected_row['Percentile']:.0f}",
                    "Percentile", "purple",
                ),
                unsafe_allow_html=True,
            )
            risk_color = {
                "High": "red", "Medium": "amber", "Low": "green",
            }.get(str(selected_row["At_Risk"]), "green")
            m4.markdown(
                kpi_card(
                    "\U0001f6a8", str(selected_row["At_Risk"]),
                    "Risk Level", risk_color,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            individual_scores = student_subject_scores.loc[
                student_subject_scores["_student"] == selected_student
            ].iloc[0].drop(labels="_student")
            scores_clean = individual_scores.dropna()

            # Score breakdown bar chart
            st.markdown(
                section_header("📊", "Score Breakdown"),
                unsafe_allow_html=True,
            )
            fig_bar = px.bar(
                x=scores_clean.index.tolist(),
                y=scores_clean.values.tolist(),
                text=[f"{v:.1f}" for v in scores_clean.values],
                color=scores_clean.values.tolist(),
                color_continuous_scale=["#F87171", "#FBBF24", "#34D399"],
                labels={"x": "Subject", "y": "Score", "color": "Score"},
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.add_hline(
                y=support_cutoff, line_dash="dash", line_color="#F87171",
                annotation_text="Support cutoff",
                annotation_font_color="#F87171",
            )
            fig_bar.add_hline(
                y=high_cutoff, line_dash="dash", line_color="#34D399",
                annotation_text="Strong cutoff",
                annotation_font_color="#34D399",
            )
            fig_bar.update_layout(coloraxis_showscale=False)
            apply_plotly_theme(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)


            st.info(
                student_advice(individual_scores, float(selected_row["Overall"]))
            )

            st.download_button(
                f"Download {selected_student}'s report",
                to_csv_bytes(
                    scores_clean.rename("Score")
                    .to_frame()
                    .reset_index()
                    .rename(columns={"index": "Subject"})
                ),
                file_name=f"edupulse_{selected_student.replace(' ', '_')}.csv",
                mime="text/csv",
            )


elif page == "Compare Students":

    st.markdown(
        section_header("⚖️", "Compare Students"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Side-by-side comparison of two or more students across all score fields."
    )

    if student_summary.empty:
        st.info("No students match the current filters.")
    else:
        compare_list = st.multiselect(
            "Select students to compare",
            student_summary["_student"].tolist(),
            default=student_summary["_student"].tolist()[:2],
            key="compare_students",
        )

        if len(compare_list) < 2:
            st.warning("Select at least two students to compare.")
        else:
            compare_df = student_summary[
                student_summary["_student"].isin(compare_list)
            ].copy()

            # KPI row
            cols = st.columns(min(len(compare_list), 6))
            for idx, (_, row) in enumerate(compare_df.iterrows()):
                col = cols[idx % len(cols)]
                band = str(row["Performance"])
                band_color = {
                    "Strong": "green",
                    "On track": "blue",
                    "Needs support": "red",
                }.get(band, "blue")
                col.markdown(
                    kpi_card(
                        "\U0001f393",
                        f"{row['Overall']:.1f}",
                        str(row["_student"])[:20],
                        band_color,
                        band,
                        "up" if band == "Strong"
                        else ("down" if band == "Needs support" else "neutral"),
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Grouped bar chart
            st.markdown(
                section_header("\U0001f4ca", "Score Comparison"),
                unsafe_allow_html=True,
            )
            compare_long = []
            for name in compare_list:
                row_data = student_subject_scores.loc[
                    student_subject_scores["_student"] == name
                ].iloc[0].drop(labels="_student")
                for subj, val in row_data.items():
                    if pd.notna(val):
                        compare_long.append(
                            {"Student": name, "Subject": subj, "Score": float(val)}
                        )
            compare_long_df = pd.DataFrame(compare_long)
            if not compare_long_df.empty:
                fig_comp = px.bar(
                    compare_long_df,
                    x="Subject",
                    y="Score",
                    color="Student",
                    barmode="group",
                    text="Score",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_comp.update_traces(
                    texttemplate="%{text:.1f}", textposition="outside"
                )
                apply_plotly_theme(fig_comp)
                st.plotly_chart(fig_comp, use_container_width=True)

            # Comparison table
            st.markdown(
                section_header("\U0001f4dd", "Comparison Table"),
                unsafe_allow_html=True,
            )
            st.dataframe(
                display_df(compare_df),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download comparison report",
                to_csv_bytes(display_df(compare_df)),
                file_name="edupulse_comparison.csv",
                mime="text/csv",
            )


elif page == "Student Support":

    st.markdown(
        section_header("\U0001fa7a", "Student Support"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Every student gets individual next steps based on their strengths and "
        "the score areas where they can grow. Study hours are not used."
    )

    if student_summary.empty:
        st.info("No students match the current filters.")
    else:
        support_students = student_summary.sort_values(
            "Overall", ascending=True
        ).copy()

        # --- AT-RISK EARLY WARNING ---
        high_risk = support_students[support_students["At_Risk"] == "High"]
        medium_risk = support_students[support_students["At_Risk"] == "Medium"]

        if not high_risk.empty or not medium_risk.empty:
            st.markdown(
                section_header("\U0001f6a8", "Early Warning"),
                unsafe_allow_html=True,
            )
            st.caption("Students currently below or near the support cutoff.")
            warn_cols = st.columns(2)
            with warn_cols[0]:
                st.markdown(
                    f'<span class="risk-badge high">HIGH RISK</span> '
                    f'<span style="color:#94A3B8; font-size:13px;"> '
                    f'{len(high_risk)} student{"s" if len(high_risk) != 1 else ""} '
                    f'below {support_cutoff:g}</span>',
                    unsafe_allow_html=True,
                )
                if not high_risk.empty:
                    st.dataframe(
                        display_df(
                            high_risk[
                                ["_student", "Overall", "Performance", "Percentile"]
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            with warn_cols[1]:
                st.markdown(
                    f'<span class="risk-badge medium">AT RISK</span> '
                    f'<span style="color:#94A3B8; font-size:13px;"> '
                    f'{len(medium_risk)} student{"s" if len(medium_risk) != 1 else ""} '
                    f'within {at_risk_margin:g} of cutoff</span>',
                    unsafe_allow_html=True,
                )
                if not medium_risk.empty:
                    st.dataframe(
                        display_df(
                            medium_risk[
                                ["_student", "Overall", "Performance", "Percentile"]
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            st.markdown("---")

        st.dataframe(
            display_df(support_students),
            use_container_width=True,
            hide_index=True,
        )

        if not support_students.empty:
            selected_student = st.selectbox(
                "View an individual student's advice",
                support_students["_student"].tolist(),
            )
            selected_row = support_students.loc[
                support_students["_student"] == selected_student
            ].iloc[0]
            student_scores_individual = student_subject_scores.loc[
                student_subject_scores["_student"] == selected_student
            ].iloc[0].drop(labels="_student")

            band = str(selected_row["Performance"])
            band_color = {
                "Strong": "green",
                "On track": "blue",
                "Needs support": "red",
            }.get(band, "blue")

            st.markdown(f"#### {selected_student}")
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                kpi_card(
                    "\U0001f3af", f"{selected_row['Overall']:.1f}",
                    "Overall", band_color,
                ),
                unsafe_allow_html=True,
            )
            m2.markdown(
                kpi_card("\U0001f3c5", band, "Band", band_color),
                unsafe_allow_html=True,
            )
            risk_c = {
                "High": "red", "Medium": "amber", "Low": "green",
            }.get(str(selected_row["At_Risk"]), "green")
            m3.markdown(
                kpi_card(
                    "\U0001f6a8",
                    str(selected_row["At_Risk"]) + " Risk",
                    "Early Warning",
                    risk_c,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            scores_clean = student_scores_individual.dropna()
            fig_support = px.bar(
                x=scores_clean.index.tolist(),
                y=scores_clean.values.tolist(),
                text=[f"{v:.1f}" for v in scores_clean.values],
                color=scores_clean.values.tolist(),
                color_continuous_scale=["#F87171", "#FBBF24", "#34D399"],
                labels={"x": "Subject", "y": "Score", "color": "Score"},
            )
            fig_support.update_traces(textposition="outside")
            fig_support.add_hline(
                y=support_cutoff, line_dash="dash", line_color="#F87171",
                annotation_text="Support cutoff",
                annotation_font_color="#F87171",
            )
            fig_support.add_hline(
                y=high_cutoff, line_dash="dash", line_color="#34D399",
                annotation_text="Strong cutoff",
                annotation_font_color="#34D399",
            )
            fig_support.update_layout(coloraxis_showscale=False)
            apply_plotly_theme(fig_support)
            st.plotly_chart(fig_support, use_container_width=True)

            st.info(str(selected_row["Personal advice"]))

        st.download_button(
            "Download support report",
            to_csv_bytes(display_df(support_students)),
            file_name="edupulse_support_report.csv",
            mime="text/csv",
        )


elif page == "Academic Insights":

    st.markdown(
        section_header("\U0001f9ea", "Academic Insights"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Spot classroom-wide patterns, correlations, and gaps between "
        "subjects / score fields."
    )

    if filtered_scores.empty:
        st.info("No students match the current filters.")
    else:
        # --- Subject comparison ---
        subject_summary = (
            filtered_scores.groupby("Subject", observed=False)["Score"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={
                "mean": "Average",
                "median": "Median",
                "std": "Std Dev",
                "count": "Results",
            })
            .sort_values("Average")
        )
        subject_summary["Std Dev"] = subject_summary["Std Dev"].fillna(0).round(2)

        st.markdown(
            section_header("\U0001f4da", "Subject / Score-Field Comparison"),
            unsafe_allow_html=True,
        )
        fig_subj = px.bar(
            subject_summary.reset_index(),
            x="Average",
            y="Subject",
            orientation="h",
            text="Average",
            color="Average",
            color_continuous_scale=["#F87171", "#FBBF24", "#34D399"],
            hover_data=["Median", "Std Dev", "Results"],
        )
        fig_subj.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_subj.update_layout(coloraxis_showscale=False)
        apply_plotly_theme(fig_subj)
        st.plotly_chart(fig_subj, use_container_width=True)
        st.dataframe(subject_summary, use_container_width=True)

        # --- Box plot distribution ---
        st.markdown(
            section_header("\U0001f4c0", "Score Distribution by Subject"),
            unsafe_allow_html=True,
        )
        fig_box = px.box(
            filtered_scores,
            x="Subject",
            y="Score",
            color="Subject",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            points="all",
        )
        fig_box.update_layout(showlegend=False)
        fig_box.add_hline(
            y=support_cutoff, line_dash="dash", line_color="#F87171",
            annotation_text="Support cutoff",
            annotation_font_color="#F87171",
        )
        fig_box.add_hline(
            y=high_cutoff, line_dash="dash", line_color="#34D399",
            annotation_text="Strong cutoff",
            annotation_font_color="#34D399",
        )
        apply_plotly_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

        # --- Correlation heatmap ---
        numeric_for_corr = student_subject_scores.drop(
            columns=["_student"], errors="ignore"
        )
        if numeric_for_corr.shape[1] >= 2:
            st.markdown(
                section_header("\U0001f517", "Correlation Heatmap"),
                unsafe_allow_html=True,
            )
            st.caption(
                "Shows how score fields relate to each other. High correlation "
                "means students tend to perform similarly across those fields."
            )
            corr_matrix = numeric_for_corr.corr()
            fig_heat = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale=["#1E293B", "#38BDF8", "#F472B6"],
                aspect="auto",
            )
            fig_heat.update_layout(
                coloraxis_colorbar=dict(
                    title="Corr",
                    tickfont=dict(color="#94A3B8"),
                ),
            )
            apply_plotly_theme(fig_heat)
            st.plotly_chart(fig_heat, use_container_width=True)

        # --- Consistency / spread analysis ---
        st.markdown(
            section_header("\U0001f3af", "Consistency Analysis"),
            unsafe_allow_html=True,
        )
        st.caption(
            "Students with high standard deviation perform inconsistently "
            "across subjects. Those near the top-right are strong but uneven."
        )
        scatter_df = student_summary[
            ["_student", "Overall", "Std_Dev", "Performance"]
        ].copy()
        color_map_scatter = {
            "Strong": "#34D399",
            "On track": "#38BDF8",
            "Needs support": "#F87171",
        }
        fig_scatter = px.scatter(
            scatter_df,
            x="Overall",
            y="Std_Dev",
            color="Performance",
            color_discrete_map=color_map_scatter,
            hover_data=["_student"],
            labels={
                "Overall": "Overall Score",
                "Std_Dev": "Standard Deviation",
                "_student": "Student",
            },
            size="Std_Dev",
            size_max=14,
        )
        fig_scatter.add_vline(
            x=support_cutoff, line_dash="dash", line_color="#F87171", opacity=0.5,
        )
        fig_scatter.add_vline(
            x=high_cutoff, line_dash="dash", line_color="#34D399", opacity=0.5,
        )
        apply_plotly_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Percentile rankings ---
        st.markdown(
            section_header("\U0001f3c6", "Percentile Rankings"),
            unsafe_allow_html=True,
        )
        st.caption("Each student's position relative to the entire class.")
        percentile_df = (
            student_summary[
                ["_student", "Overall", "Percentile", "Performance"]
            ]
            .sort_values("Percentile", ascending=False)
            .copy()
        )
        fig_pct = px.bar(
            percentile_df,
            x="_student",
            y="Percentile",
            color="Performance",
            color_discrete_map=color_map_scatter,
            hover_data=["Overall"],
            labels={"_student": "Student", "Percentile": "Percentile Rank"},
        )
        fig_pct.update_layout(xaxis_tickangle=-45)
        apply_plotly_theme(fig_pct)
        st.plotly_chart(fig_pct, use_container_width=True)

        st.caption(
            f"Performance bands: Needs support < {support_cutoff:g}, "
            f"On track < {high_cutoff:g}, Strong ≥ {high_cutoff:g} "
            f"(scale maximum {score_scale:g})."
        )

        st.download_button(
            "Download insights report",
            to_csv_bytes(subject_summary.reset_index()),
            file_name="edupulse_academic_insights.csv",
            mime="text/csv",
        )
