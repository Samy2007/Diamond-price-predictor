import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Estimator", page_icon="💎")

FEATS = ["cut", "color", "clarity", "carat", "x", "y", "z"]
CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLORS = list("DEFGHIJ")  # D = best, J = worst
CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # IF = best

# Typical error on the held-out test set (MAE, USD). Update from your own results.
TEST_MAE = 268


@st.cache_resource
def load_assets():
    model = joblib.load("diamond_model.joblib")
    with open("input_ranges.json") as f:
        ranges = json.load(f)
    return model, ranges


model, R = load_assets()

st.title("💎 Diamond Price Estimator")
st.write(
    "Estimate a fair market price for a diamond from its grade and measurements "
    "(all values are printed on a grading certificate)."
)

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat (weight)", min_value=0.0, value=0.90, step=0.01)
    cut = st.selectbox("Cut", CUTS, index=4)
    color = st.selectbox("Color (D = best)", COLORS, index=3)
    clarity = st.selectbox("Clarity (IF = best)", CLARITIES, index=3)
with col2:
    x = st.number_input("Length x (mm)", min_value=0.0, value=6.20, step=0.01)
    y = st.number_input("Width y (mm)", min_value=0.0, value=6.20, step=0.01)
    z = st.number_input("Depth z (mm)", min_value=0.0, value=3.80, step=0.01)

if st.button("Estimate price", type="primary"):
    values = {"carat": carat, "x": x, "y": y, "z": z}

    # 1) Basic validation: everything must be positive
    if min(values.values()) <= 0:
        st.error("Carat and all dimensions must be greater than zero.")
        st.stop()

    # 2) Range validation: values must be inside the range the model was trained on
    problems = []
    for name, val in values.items():
        lo, hi = R[name]["min"], R[name]["max"]
        if not (lo <= val <= hi):
            problems.append(f"{name} = {val} is outside the supported range ({lo:g} to {hi:g}).")
    if problems:
        st.error("Some values are outside what this model was trained on:\n\n- " + "\n- ".join(problems))
        st.stop()

    # 3) Consistency check: carat vs dimensions (soft warning)
    ratio = carat / (x * y * z)
    low_confidence = not (R["_ratio"]["ratio_lo"] <= ratio <= R["_ratio"]["ratio_hi"])
    if low_confidence:
        st.error(
            "The carat weight does not match the dimensions you entered, "
            "so no estimate is shown. Please double-check the numbers on the certificate."
        )
        st.stop()

    # 4) Predict
    try:
        row = pd.DataFrame([{"cut": cut, "color": color, "clarity": clarity,
                             "carat": carat, "x": x, "y": y, "z": z}])
        price = float(model.predict(row[FEATS])[0])
    except Exception as e:
        st.error(f"Could not compute a prediction: {e}")
        st.stop()

    st.metric("Estimated price", f"${price:,.0f}")
    st.caption(
        f"On diamonds the model had never seen, its average error was about ${TEST_MAE}. "
        "Errors are larger for stones above 2 carat, so review those by hand."
    )

st.divider()
st.caption(
    "Model: gradient boosting trained on the Diamonds dataset (53,940 diamonds, from the R package ggplot2; "
    "Kaggle copy by shivam2503). This is an estimate, not an appraisal."
)
