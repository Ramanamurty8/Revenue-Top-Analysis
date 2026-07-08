from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st
from scipy.stats import norm


st.set_page_config(
    page_title="Revenue Analysis",
    layout="wide",
)

st.title("Revenue and Time-on-Page Dashboard")
st.caption("Revenue analysis, regression models, and distribution explorer.")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

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

    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data["top"] = pd.to_numeric(data["top"], errors="coerce")

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


# --------------------------------------------------
# Main navigation
# --------------------------------------------------

main_page = st.sidebar.radio(
    "Choose section",
    [
        "Q1 Dataset Dashboard",
        "Q2 Distribution Explorer",
    ],
    key="main_page",
)


# --------------------------------------------------
# Q1 Dataset Dashboard
# --------------------------------------------------

if main_page == "Q1 Dataset Dashboard":

    data_file = Path(__file__).resolve().parent / "testdata.csv"

    try:
        df = load_data(data_file)
    except Exception as exc:
        st.error(f"Could not load {data_file}: {exc}")
        st.stop()

    st.sidebar.header("Dataset Filters")

    filtered = df.copy()

    for column in ["browser", "platform", "site"]:
        choices = sorted(df[column].unique())

        selected = st.sidebar.multiselect(
            column.title(),
            choices,
            default=choices,
        )

        filtered = filtered[
            filtered[column].isin(selected)
        ]

    if filtered.empty:
        st.warning("The selected filters contain no rows.")
        st.stop()

    page = st.radio(
        "Choose dashboard page",
        [
            "Overview",
            "Revenue vs TOP",
            "Segments",
            "Regression",
            "Data",
        ],
        horizontal=True,
        key="q1_page",
    )

    # --------------------------------------------------
    # Overview
    # --------------------------------------------------

    if page == "Overview":
        st.header("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", f"{len(filtered):,}")
        c2.metric("Total revenue", f"{filtered['revenue'].sum():,.2f}")
        c3.metric("Median revenue", f"{filtered['revenue'].median():,.2f}")
        c4.metric("Median TOP", f"{filtered['top'].median():,.2f}")

        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots()

            sns.histplot(
                filtered["revenue"],
                bins=30,
                kde=True,
                ax=ax,
            )

            ax.set_title("Revenue Distribution")
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Frequency")

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

            ax.set_title("TOP Distribution")
            ax.set_xlabel("TOP")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Descriptive Statistics")

        st.dataframe(
            filtered[["revenue", "top"]].describe(),
            use_container_width=True,
        )

    # --------------------------------------------------
    # Revenue vs TOP
    # --------------------------------------------------

    elif page == "Revenue vs TOP":
        st.header("Revenue versus TOP")

        hue = st.selectbox(
            "Color points by",
            [
                "None",
                "browser",
                "platform",
                "site",
            ],
            key="relationship_hue",
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            data=filtered,
            x="top",
            y="revenue",
            hue=None if hue == "None" else hue,
            alpha=0.65,
            ax=ax,
        )

        ax.set_title("Revenue versus TOP")
        ax.set_xlabel("TOP")
        ax.set_ylabel("Revenue")

        st.pyplot(fig)
        plt.close(fig)

        try:
            deciles = pd.qcut(
                filtered["top"],
                q=10,
                duplicates="drop",
            )

            summary = (
                filtered.assign(top_bin=deciles)
                .groupby("top_bin", observed=True)
                .agg(
                    median_top=("top", "median"),
                    median_revenue=("revenue", "median"),
                    count=("revenue", "size"),
                )
                .reset_index(drop=True)
            )

            summary["revenue_change_pct"] = (
                summary["median_revenue"]
                .pct_change()
                .mul(100)
            )

            st.subheader("Median Revenue by TOP Decile")

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
                "There are not enough distinct TOP values to create deciles."
            )

    # --------------------------------------------------
    # Segments
    # --------------------------------------------------

    elif page == "Segments":
        st.header("Segment Analysis")

        group = st.selectbox(
            "Compare by",
            [
                "browser",
                "platform",
                "site",
            ],
            key="segment_group",
        )

        metric = st.radio(
            "Metric",
            [
                "revenue",
                "top",
            ],
            horizontal=True,
            key="segment_metric",
        )

        left, right = st.columns([2, 1])

        with left:
            fig, ax = plt.subplots(figsize=(10, 5))

            sns.boxplot(
                data=filtered,
                x=group,
                y=metric,
                ax=ax,
            )

            ax.tick_params(axis="x", rotation=35)
            ax.set_title(f"{metric.title()} by {group.title()}")

            st.pyplot(fig)
            plt.close(fig)

        with right:
            distribution = (
                filtered[group]
                .value_counts(normalize=True)
                .mul(100)
                .rename("Percent")
            )

            st.bar_chart(distribution)

        segment_summary = (
            filtered.groupby(group)[["top", "revenue"]]
            .agg(["count", "mean", "median", "std"])
        )

        st.subheader("Segment Summary")

        st.dataframe(
            segment_summary.round(2),
            use_container_width=True,
        )

    # --------------------------------------------------
    # Regression
    # --------------------------------------------------

    elif page == "Regression":
        st.header("Regression Analysis")

        model_name = st.selectbox(
            "Select model",
            [
                "TOP only",
                "TOP + segment effects",
                "TOP interactions with each segment",
                "TOP x browser x platform",
                "TOP x browser x platform x site",
            ],
            key="regression_model",
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

        selected_formula = formulas[model_name]

        st.write("Model formula:")
        st.code(selected_formula)

        st.info(
            "The full interaction model includes browser, platform, site, TOP, "
            "and their combined interaction effects."
        )

        try:
            model = smf.ols(
                selected_formula,
                data=filtered,
            ).fit(cov_type="HC3")

            c1, c2, c3 = st.columns(3)

            c1.metric("R-squared", f"{model.rsquared:.3f}")
            c2.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
            c3.metric("Observations", f"{int(model.nobs):,}")

            confidence_intervals = model.conf_int()

            coefficient_table = pd.DataFrame(
                {
                    "coefficient": model.params,
                    "std_error": model.bse,
                    "p_value": model.pvalues,
                    "ci_low": confidence_intervals[0],
                    "ci_high": confidence_intervals[1],
                }
            )

            st.subheader("Coefficients")

            st.dataframe(
                coefficient_table.round(4),
                use_container_width=True,
            )

            diagnostics = filtered.copy()
            diagnostics["predicted"] = model.fittedvalues
            diagnostics["residual"] = model.resid

            fig, ax = plt.subplots(figsize=(9, 4))

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

            ax.set_title("Residuals versus Fitted Values")
            ax.set_xlabel("Predicted Revenue")
            ax.set_ylabel("Residual")

            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Full Model Summary"):
                st.text(model.summary().as_text())

        except Exception as exc:
            st.error(
                "The model could not be estimated: "
                f"{exc}"
            )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    elif page == "Data":
        st.header("Filtered Dataset")

        st.dataframe(
            filtered,
            use_container_width=True,
        )

        st.download_button(
            "Download filtered data",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# Q2 Distribution Explorer
# --------------------------------------------------

elif main_page == "Q2 Distribution Explorer":

    st.header("Normal Distribution Explorer")

    st.write(
        "Explore the normal curve, empirical rule, Chebyshev's inequality, "
        "random samples, and z-scores."
    )

    with st.expander("Distribution Parameters", expanded=True):
        p1, p2 = st.columns(2)
        p3, p4 = st.columns(2)

        mu = p1.slider(
            "Mean",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            key="distribution_mean",
        )

        sigma = p2.slider(
            "Standard deviation",
            min_value=0.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="distribution_std",
        )

        sample_size = p3.slider(
            "Sample size",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            key="distribution_sample_size",
        )

        k = p4.slider(
            "k for Chebyshev",
            min_value=1.1,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key="distribution_k",
        )

        show_sample = st.checkbox(
            "Show random sample histogram",
            value=True,
            key="distribution_show_sample",
        )

    x = np.linspace(
        mu - 4 * sigma,
        mu + 4 * sigma,
        1000,
    )

    pdf = norm.pdf(
        x,
        mu,
        sigma,
    )

    sample = (
        np.random.default_rng(42)
        .normal(
            mu,
            sigma,
            sample_size,
        )
    )

    st.subheader("Normal Distribution Curve")

    normal_fig = go.Figure()

    normal_fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Normal PDF",
        )
    )

    normal_fig.add_vline(
        x=mu,
        line_dash="dash",
        annotation_text="Mean",
    )

    normal_fig.add_vline(
        x=mu - sigma,
        line_dash="dot",
        annotation_text="Mean - 1 SD",
    )

    normal_fig.add_vline(
        x=mu + sigma,
        line_dash="dot",
        annotation_text="Mean + 1 SD",
    )

    normal_fig.add_vline(
        x=mu - 2 * sigma,
        line_dash="dot",
        annotation_text="Mean - 2 SD",
    )

    normal_fig.add_vline(
        x=mu + 2 * sigma,
        line_dash="dot",
        annotation_text="Mean + 2 SD",
    )

    normal_fig.update_layout(
        title="Normal Distribution PDF",
        xaxis_title="X value",
        yaxis_title="Density",
        height=500,
    )

    st.plotly_chart(
        normal_fig,
        use_container_width=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Mean", f"{mu:.2f}")
    m2.metric("Standard deviation", f"{sigma:.2f}")
    m3.metric("Variance", f"{sigma ** 2:.2f}")
    m4.metric("Sample size", f"{sample_size:,}")

    st.subheader("Empirical Rule")

    st.write(
        """
        For a normal distribution:

        - Approximately 68% of values fall within 1 standard deviation.
        - Approximately 95% fall within 2 standard deviations.
        - Approximately 99.7% fall within 3 standard deviations.
        """
    )

    empirical_data = pd.DataFrame(
        {
            "Range": [
                "Mean +/- 1 SD",
                "Mean +/- 2 SD",
                "Mean +/- 3 SD",
            ],
            "Lower bound": [
                mu - sigma,
                mu - 2 * sigma,
                mu - 3 * sigma,
            ],
            "Upper bound": [
                mu + sigma,
                mu + 2 * sigma,
                mu + 3 * sigma,
            ],
            "Approximate coverage": [
                "68%",
                "95%",
                "99.7%",
            ],
        }
    )

    st.dataframe(
        empirical_data,
        use_container_width=True,
    )

    st.subheader("Chebyshev's Inequality")

    chebyshev_min = 1 - (1 / k**2)

    normal_actual = (
        norm.cdf(k)
        - norm.cdf(-k)
    )

    st.write(
        f"For any distribution, at least "
        f"{chebyshev_min * 100:.2f}% "
        f"of values lie within {k:.1f} "
        f"standard deviations of the mean."
    )

    st.write(
        f"For a normal distribution, the actual percentage is approximately "
        f"{normal_actual * 100:.2f}%."
    )

    comparison = pd.DataFrame(
        {
            "Method": [
                "Chebyshev minimum",
                "Actual normal distribution",
            ],
            "Percentage": [
                chebyshev_min * 100,
                normal_actual * 100,
            ],
        }
    )

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
    )

    lower_k = mu - k * sigma
    upper_k = mu + k * sigma

    x_fill = x[
        (x >= lower_k)
        & (x <= upper_k)
    ]

    y_fill = norm.pdf(
        x_fill,
        mu,
        sigma,
    )

    chebyshev_fig = go.Figure()

    chebyshev_fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Normal PDF",
        )
    )

    chebyshev_fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [
                    x_fill,
                    x_fill[::-1],
                ]
            ),
            y=np.concatenate(
                [
                    y_fill,
                    np.zeros_like(y_fill),
                ]
            ),
            fill="toself",
            name=f"Within {k:.1f} SD",
        )
    )

    chebyshev_fig.add_vline(
        x=lower_k,
        line_dash="dash",
    )

    chebyshev_fig.add_vline(
        x=upper_k,
        line_dash="dash",
    )

    chebyshev_fig.update_layout(
        title="Chebyshev Interval",
        xaxis_title="X value",
        yaxis_title="Density",
        height=500,
    )

    st.plotly_chart(
        chebyshev_fig,
        use_container_width=True,
    )

    if show_sample:
        st.subheader("Random Sample Histogram")

        histogram_fig = go.Figure()

        histogram_fig.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=40,
                histnorm="probability density",
                name="Random sample",
            )
        )

        histogram_fig.add_trace(
            go.Scatter(
                x=x,
                y=pdf,
                mode="lines",
                name="Theoretical normal curve",
            )
        )

        histogram_fig.update_layout(
            title="Random Sample versus Theoretical Normal Curve",
            xaxis_title="Value",
            yaxis_title="Density",
            height=500,
        )

        st.plotly_chart(
            histogram_fig,
            use_container_width=True,
        )

    st.subheader("Z-Score Calculator")

    value = st.number_input(
        "Enter a value X",
        value=float(mu),
        key="distribution_value",
    )

    z_score = (value - mu) / sigma

    percentile = norm.cdf(z_score) * 100

    z1, z2 = st.columns(2)

    z1.metric("Z-score", f"{z_score:.3f}")
    z2.metric("Percentile", f"{percentile:.2f}%")

    st.write(
        f"A value of {value} is "
        f"{z_score:.3f} standard deviations from the mean."
    )

    with st.expander("Assumptions and Interpretation"):
        st.write(
            """
            The normal distribution is most appropriate when:

            - The data is approximately symmetric.
            - Most values are close to the mean.
            - Extreme values are uncommon.
            - Mean, median, and mode are close.
            - The variable is continuous.
            - Many small independent effects contribute to the result.

            Revenue, income, spending, clicks, and waiting times are often
            skewed. Check histograms, boxplots, skewness, and outliers before
            applying methods that assume normality.
            """
        )