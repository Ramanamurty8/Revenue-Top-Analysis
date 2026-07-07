import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# Page Config

st.set_page_config(
    page_title="Normal Distribution Explorer",
    layout="wide"
)

st.title("📊 Normal Distribution Explorer")
st.write(
    """
    This dashboard helps you understand the Normal Distribution, 
    empirical rule, Chebyshev's Inequality, z-scores, and assumptions.
    """
)


# Sidebar Inputs

st.sidebar.header("Distribution Parameters")

mu = st.sidebar.slider("Mean (μ)", -50.0, 50.0, 0.0, 1.0)
sigma = st.sidebar.slider("Standard Deviation (σ)", 0.5, 20.0, 5.0, 0.5)
sample_size = st.sidebar.slider("Sample Size", 100, 10000, 1000, 100)

k = st.sidebar.slider(
    "k for Chebyshev's Inequality",
    1.1, 5.0, 2.0, 0.1
)

show_sample = st.sidebar.checkbox("Show Random Sample Histogram", True)


# Generate Normal Distribution

x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
pdf = norm.pdf(x, mu, sigma)

sample = np.random.normal(mu, sigma, sample_size)


# Main Distribution Plot

st.subheader("1. Normal Distribution Curve")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=pdf,
        mode="lines",
        name="Normal PDF"
    )
)

fig.add_vline(x=mu, line_dash="dash", annotation_text="Mean μ")

fig.add_vline(x=mu - sigma, line_dash="dot", annotation_text="μ - 1σ")
fig.add_vline(x=mu + sigma, line_dash="dot", annotation_text="μ + 1σ")

fig.add_vline(x=mu - 2 * sigma, line_dash="dot", annotation_text="μ - 2σ")
fig.add_vline(x=mu + 2 * sigma, line_dash="dot", annotation_text="μ + 2σ")

fig.update_layout(
    title="Normal Distribution PDF",
    xaxis_title="X value",
    yaxis_title="Density",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# Summary Statistics

st.subheader("2. Distribution Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Mean", round(mu, 2))
col2.metric("Standard Deviation", round(sigma, 2))
col3.metric("Variance", round(sigma ** 2, 2))
col4.metric("Sample Size", sample_size)


# Empirical Rule

st.subheader("3. Empirical Rule")

st.write(
    """
    For a perfectly normal distribution:
    - About **68%** of values lie within **1 standard deviation**
    - About **95%** of values lie within **2 standard deviations**
    - About **99.7%** of values lie within **3 standard deviations**
    """
)

empirical_data = pd.DataFrame({
    "Range": [
        "μ ± 1σ",
        "μ ± 2σ",
        "μ ± 3σ"
    ],
    "Lower Bound": [
        mu - sigma,
        mu - 2 * sigma,
        mu - 3 * sigma
    ],
    "Upper Bound": [
        mu + sigma,
        mu + 2 * sigma,
        mu + 3 * sigma
    ],
    "Approx % Covered": [
        "68%",
        "95%",
        "99.7%"
    ]
})

st.dataframe(empirical_data, use_container_width=True)


# Chebyshev's Inequality

st.subheader("4. Chebyshev's Inequality")

chebyshev_min = 1 - (1 / k ** 2)
normal_actual = norm.cdf(k) - norm.cdf(-k)

st.write(
    f"""
    Chebyshev's Inequality says that for **any distribution**, not just normal:

    At least **{round(chebyshev_min * 100, 2)}%** of values lie within **{k} standard deviations** of the mean.

    For a normal distribution, the actual percentage within **{k}σ** is approximately 
    **{round(normal_actual * 100, 2)}%**.
    """
)

cheb_df = pd.DataFrame({
    "Method": ["Chebyshev Minimum Guarantee", "Actual Normal Distribution"],
    "Percentage Within k Standard Deviations": [
        round(chebyshev_min * 100, 2),
        round(normal_actual * 100, 2)
    ]
})

st.dataframe(cheb_df, use_container_width=True)


# Chebyshev Plot

lower_k = mu - k * sigma
upper_k = mu + k * sigma

fig_cheb = go.Figure()

fig_cheb.add_trace(
    go.Scatter(
        x=x,
        y=pdf,
        mode="lines",
        name="Normal PDF"
    )
)

x_fill = x[(x >= lower_k) & (x <= upper_k)]
y_fill = norm.pdf(x_fill, mu, sigma)

fig_cheb.add_trace(
    go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
        fill="toself",
        name=f"Within {k}σ"
    )
)

fig_cheb.add_vline(x=lower_k, line_dash="dash", annotation_text=f"μ - {k}σ")
fig_cheb.add_vline(x=upper_k, line_dash="dash", annotation_text=f"μ + {k}σ")

fig_cheb.update_layout(
    title="Chebyshev Interval on Normal Distribution",
    xaxis_title="X value",
    yaxis_title="Density",
    height=500
)

st.plotly_chart(fig_cheb, use_container_width=True)


# Random Sample Histogram

if show_sample:
    st.subheader("5. Random Sample Histogram")

    hist_fig = go.Figure()

    hist_fig.add_trace(
        go.Histogram(
            x=sample,
            nbinsx=40,
            histnorm="probability density",
            name="Random Sample"
        )
    )

    hist_fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Theoretical Normal Curve"
        )
    )

    hist_fig.update_layout(
        title="Random Sample vs Theoretical Normal Curve",
        xaxis_title="Value",
        yaxis_title="Density",
        height=500
    )

    st.plotly_chart(hist_fig, use_container_width=True)


# Z-score Calculator

st.subheader("6. Z-Score Calculator")

value = st.number_input("Enter a value X", value=float(mu))

z_score = (value - mu) / sigma
percentile = norm.cdf(z_score) * 100

col5, col6 = st.columns(2)

col5.metric("Z-score", round(z_score, 3))
col6.metric("Percentile", f"{round(percentile, 2)}%")

st.write(
    f"""
    A value of **{value}** is **{round(z_score, 3)} standard deviations**
    away from the mean.
    """
)


# Assumptions Section

st.subheader("7. Assumptions and When Normal Distribution Makes Sense")

st.write(
    """
    The normal distribution is a good assumption when:

    - Data is roughly **symmetric**
    - Most values are close to the mean
    - Extreme values are rare
    - Mean, median, and mode are close
    - The variable is continuous
    - Data comes from many small independent effects added together

    Examples where normal distribution may work:
    - Heights
    - Measurement errors
    - Test scores
    - Manufacturing variation

    Examples where normal distribution may NOT work:
    - Revenue
    - Income
    - Website clicks
    - Waiting time
    - Customer spending
    - Highly skewed business metrics
    """
)


# Normality Warning

st.subheader("8. Important Insight")

st.info(
    """
    Do not blindly assume normality. In real business data, many metrics like revenue,
    session duration, spend, and clicks are usually skewed. Always check histogram,
    boxplot, skewness, and outliers before applying normal-based methods.
    """
)


