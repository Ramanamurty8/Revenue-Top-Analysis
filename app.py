from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st


st.set_page_config(
    page_title="Revenue Analysis",
    layout="wide",
)

st.title("Revenue and Time-on-Page Dashboard")
st.caption(
    "Explore revenue, TOP, audience segments, and regression results."
)


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    required = {
        "revenue",
        "top",
        "browser",
        "platform",
        "site",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.dropna(
        subset=["revenue", "top"]
    ).copy()

    for column in ["browser", "platform", "site"]:
        data[column] = (
            data[column]
            .fillna("Missing")
            .astype(str)
        )

    return data


# Load testdata.csv from the same folder as app.py
data_file = (
    Path(__file__).resolve().parent
    / "testdata.csv"
)

try:
    df = load_data(data_file)
except Exception as exc:
    st.error(
        f"Could not load {data_file}: {exc}"
    )
    st.stop()


st.sidebar.header("Filters")
filtered = df.copy()

for column in ["browser", "platform", "site"]:
    choices = sorted(
        df[column].unique()
    )

    selected = st.sidebar.multiselect(
        column.title(),
        choices,
        default=choices,
    )

    filtered = filtered[
        filtered[column].isin(selected)
    ]


if filtered.empty:
    st.warning(
        "The selected filters contain no rows."
    )
    st.stop()


(
    overview,
    relationship,
    segments,
    regression,
    data_tab,
) = st.tabs(
    [
        "Overview",
        "Revenue vs TOP",
        "Segments",
        "Regression",
        "Data",
    ]
)


with overview:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(filtered):,}",
    )

    c2.metric(
        "Total revenue",
        f"{filtered['revenue'].sum():,.2f}",
    )

    c3.metric(
        "Median revenue",
        f"{filtered['revenue'].median():,.2f}",
    )

    c4.metric(
        "Median TOP",
        f"{filtered['top'].median():,.2f}",
    )

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()

        sns.histplot(
            filtered["revenue"],
            bins=30,
            kde=True,
            ax=ax,
        )

        ax.set_title(
            "Revenue distribution"
        )

        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots()

        sns.histplot(
            filtered["top"],
            bins=30,
            kde=True,
            ax=ax,
        )

        ax.set_title(
            "TOP distribution"
        )

        st.pyplot(fig)
        plt.close(fig)

    st.subheader(
        "Descriptive statistics"
    )

    st.dataframe(
        filtered[
            ["revenue", "top"]
        ].describe(),
        use_container_width=True,
    )


with relationship:
    hue = st.selectbox(
        "Color points by",
        [
            "None",
            "browser",
            "platform",
            "site",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.scatterplot(
        data=filtered,
        x="top",
        y="revenue",
        hue=None if hue == "None" else hue,
        alpha=0.65,
        ax=ax,
    )

    ax.set_title("Revenue vs TOP")

    st.pyplot(fig)
    plt.close(fig)

    try:
        deciles = pd.qcut(
            filtered["top"],
            q=10,
            duplicates="drop",
        )

        summary = (
            filtered.assign(
                top_bin=deciles
            )
            .groupby(
                "top_bin",
                observed=True,
            )
            .agg(
                median_top=(
                    "top",
                    "median",
                ),
                median_revenue=(
                    "revenue",
                    "median",
                ),
                count=(
                    "revenue",
                    "size",
                ),
            )
            .reset_index(drop=True)
        )

        summary[
            "revenue_change_pct"
        ] = (
            summary["median_revenue"]
            .pct_change()
            .mul(100)
        )

        st.subheader(
            "Median revenue by TOP decile"
        )

        st.line_chart(
            summary,
            x="median_top",
            y="median_revenue",
        )

        st.dataframe(
            summary.round(2),
            use_container_width=True,
        )

    except ValueError:
        st.info(
            "There are not enough distinct "
            "TOP values to create deciles."
        )


with segments:
    group = st.selectbox(
        "Compare by",
        ["browser", "platform", "site"],
    )

    metric = st.radio(
        "Metric",
        ["revenue", "top"],
        horizontal=True,
    )

    left, right = st.columns([2, 1])

    with left:
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.boxplot(
            data=filtered,
            x=group,
            y=metric,
            ax=ax,
        )

        ax.tick_params(
            axis="x",
            rotation=35,
        )

        ax.set_title(
            f"{metric.title()} by "
            f"{group.title()}"
        )

        st.pyplot(fig)
        plt.close(fig)

    with right:
        distribution = (
            filtered[group]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .rename("percent")
        )

        st.bar_chart(distribution)

    segment_summary = (
        filtered.groupby(group)[
            ["top", "revenue"]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
    )

    st.dataframe(
        segment_summary.round(2),
        use_container_width=True,
    )


with regression:
    model_name = st.selectbox(
        "Model",
        [
            "TOP only",
            "TOP + segment effects",
            "TOP interactions with each segment",
            "TOP x browser x platform",
            "TOP x browser x platform x site",
        ],
    )

    formulas = {
        "TOP only": (
            "revenue ~ top"
        ),

        "TOP + segment effects": (
            "revenue ~ top "
            "+ C(browser) "
            "+ C(platform) "
            "+ C(site)"
        ),

        "TOP interactions with each segment": (
            "revenue ~ "
            "top*C(browser) "
            "+ top*C(platform) "
            "+ top*C(site)"
        ),

        "TOP x browser x platform": (
            "revenue ~ "
            "top*C(browser)*C(platform) "
            "+ C(site)"
        ),

        "TOP x browser x platform x site": (
            "revenue ~ "
            "top*C(browser)*C(platform)*C(site)"
        ),
    }

    selected_formula = formulas[
        model_name
    ]

    st.write("Model formula:")

    st.code(selected_formula)

    try:
        model = smf.ols(
            selected_formula,
            data=filtered,
        ).fit(cov_type="HC3")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "R-squared",
            f"{model.rsquared:.3f}",
        )

        c2.metric(
            "Adjusted R-squared",
            f"{model.rsquared_adj:.3f}",
        )

        c3.metric(
            "Observations",
            f"{int(model.nobs):,}",
        )

        confidence_intervals = (
            model.conf_int()
        )

        coefficient_table = pd.DataFrame(
            {
                "coefficient": model.params,
                "std_error": model.bse,
                "p_value": model.pvalues,
                "ci_low": (
                    confidence_intervals[0]
                ),
                "ci_high": (
                    confidence_intervals[1]
                ),
            }
        )

        st.subheader("Coefficients")

        st.dataframe(
            coefficient_table.round(4),
            use_container_width=True,
        )

        diagnostics = filtered.copy()

        diagnostics["predicted"] = (
            model.fittedvalues
        )

        diagnostics["residual"] = (
            model.resid
        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        sns.scatterplot(
            data=diagnostics,
            x="predicted",
            y="residual",
            alpha=0.65,
            ax=ax,
        )

        ax.axhline(
            0,
            color="red",
            linewidth=1,
        )

        ax.set_title(
            "Residuals vs fitted values"
        )

        st.pyplot(fig)
        plt.close(fig)

        with st.expander(
            "Full model summary"
        ):
            st.text(
                model.summary().as_text()
            )

    except Exception as exc:
        st.error(
            "The model could not be "
            f"estimated: {exc}"
        )


with data_tab:
    st.dataframe(
        filtered,
        use_container_width=True,
    )

    st.download_button(
        "Download filtered data",
        filtered.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )