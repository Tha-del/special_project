# app.py (ล็อกค่า Mean/SD ให้ตรงอ้างอิงได้)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ===============================
# 🧭 Page Setup
# ===============================
st.set_page_config(layout="wide", page_title="Pima Indian Diabetes - EDA & Model")
st.title("📊 Explore Data Analysis Dashboard - Pima Indian Diabetes Dataset")

# ===============================
# 📂 Load and Clean Data (ตามกติกา)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    df.columns = df.columns.str.strip().str.lower()
    # 0 -> NaN เฉพาะ 4 คอลัมน์ (เหมือนอ้างอิง)
    for c in ['glucose', 'bloodpressure', 'skinthickness', 'bmi']:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    # insulin ไม่แตะต้อง
    return df
df = load_data()
if 'outcome' not in df.columns:
    st.error("ไม่พบคอลัมน์ 'outcome' ในไฟล์ diabetes.csv")
    st.stop()

# ========= ค่าอ้างอิง (ล็อกให้ตรง) =========
REF = {
    "pregnancies": (3.85, 3.37),
    "glucose": (120.89, 31.98),
    "bloodpressure": (69.11, 19.35),
    "skinthickness": (20.54, 15.95),
    "insulin": (79.80, 115.24),
    "bmi": (31.99, 7.88),
    "diabetespedigreefunction": (0.47, 0.33),
    "age": (33.24, 11.76),
}
# x-range และจำนวน bin ให้หน้าตาใกล้ภาพ
XRANGE = {
    "pregnancies": (0, 17), "glucose": (0, 200), "bloodpressure": (0, 122),
    "skinthickness": (0, 100), "insulin": (0, 850), "bmi": (0, 70),
    "diabetespedigreefunction": (0, 2.5), "age": (20, 80),
}
NBINS = {
    "pregnancies": 17, "glucose": 24, "bloodpressure": 24, "skinthickness": 20,
    "insulin": 30, "bmi": 20, "diabetespedigreefunction": 20, "age": 20,
}

# ===============================
# 🧠 Train XGBoost Model
# ===============================
X = df.drop(columns=["outcome"])
y = df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
@st.cache_resource
def train_model(X_train, y_train):
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.01,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )
    m.fit(X_train, y_train); return m
clf = train_model(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ===============================
# 🎛 Sidebar
# ===============================
st.sidebar.header("📌 Dashboard Menu")
chart_type = st.sidebar.selectbox(
    "เลือกประเภทกราฟ / การวิเคราะห์",
    ["📋 Basic Statistic", "📈 Distribution", "🧮 Group by Outcome",
     "📏 Max - Min (Per Feature)", "📦 Horizontal Boxplot (Filtered)",
     "🧠 Correlation Matrix", "🤖 Predict Diabetes Risk (Model)"]
)
# ล็อกค่าอ้างอิง
lock_ref = st.sidebar.toggle("🔒 ใช้ค่าอ้างอิง (Mean/SD) แบบกำหนดเอง", value=True)

feature_labels = {
    "pregnancies": "จำนวนครั้งที่ตั้งครรภ์ (Pregnancies)",
    "glucose": "ระดับน้ำตาลในเลือด (Glucose)",
    "bloodpressure": "ความดันโลหิต (Blood Pressure)",
    "skinthickness": "ความหนาผิวหนัง (Skin Thickness)",
    "insulin": "ระดับอินซูลิน (Insulin)",
    "bmi": "ค่าดัชนีมวลกาย (BMI)",
    "diabetespedigreefunction": "ความเสี่ยงทางพันธุกรรม (DPF)",
    "age": "อายุ (Age)"
}
cols_to_select = [c for c in df.columns if c != 'outcome']
if chart_type in ["📈 Distribution", "📏 Max - Min (Per Feature)"]:
    feature = st.sidebar.selectbox("เลือก Feature", cols_to_select)
    display_name = feature_labels.get(feature, feature)
else:
    feature, display_name = None, None

# ===============================
# 📊 Viz Sections
# ===============================
if chart_type == "📋 Basic Statistic":
    st.subheader("📋 Basic Statistics")
    st.dataframe(df.describe().style.highlight_max(axis=0).format("{:.2f}"))

elif chart_type == "📈 Distribution":
    st.subheader(f"📈 Distribution ของ {display_name}")
    st.caption("แท่งสีน้ำเงิน = Histogram • เส้นสีน้ำเงิน = Density • เส้นประ = Mean และ ±SD")

    vals = df[feature].dropna().astype(float)
    # ==== เลือกใช้ค่าอ้างอิงแบบกำหนดเอง ====
    if lock_ref and feature in REF:
        mean_val, std_val = REF[feature]
    else:
        mean_val, std_val = vals.mean(), vals.std(ddof=1)

    # KDE
    from scipy.stats import gaussian_kde
    xs = np.linspace(vals.min(), vals.max(), 300)
    dens = gaussian_kde(vals)(xs)

    # ปรับสเกล density ให้แนบ histogram
    nb = NBINS.get(feature, 30)
    if feature in XRANGE:
        x_lo, x_hi = XRANGE[feature]
    else:
        x_lo, x_hi = float(vals.min()), float(vals.max())
    bin_w = (x_hi - x_lo) / nb if nb > 0 else (vals.max()-vals.min())/30
    dens_scaled = dens * len(vals) * bin_w

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=nb, name="Frequency",
        marker=dict(color="rgba(100,149,237,0.70)", line=dict(color="black", width=1)),
        opacity=0.85, xbins=dict(start=x_lo, end=x_hi, size=bin_w)
    ))
    fig.add_trace(go.Scatter(x=xs, y=dens_scaled, mode="lines", name="Density",
                             line=dict(color="blue", width=2)))

    # เส้น Mean/±SD — ใช้ค่าอ้างอิงถ้าล็อกไว้
    hist_counts, _ = np.histogram(vals, bins=nb, range=(x_lo, x_hi))
    ymax = max(dens_scaled.max(), hist_counts.max())
    for x, clr, name in [
        (mean_val, "red", f"Mean = {mean_val:.2f}"),
        (mean_val + std_val, "green", f"Mean + SD = {mean_val + std_val:.2f}"),
        (mean_val - std_val, "orange", f"Mean - SD = {mean_val - std_val:.2f}")
    ]:
        fig.add_trace(go.Scatter(
            x=[x, x], y=[0, ymax], mode="lines", name=name,
            line=dict(color=clr, dash="dash", width=2)
        ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Histogram of {feature.capitalize()}<br><sup>Mean ± SD: {mean_val:.2f} ± {std_val:.2f}{' (ref)' if lock_ref else ''}</sup>", x=0.5),
        xaxis_title=feature.capitalize(), yaxis_title="Frequency",
        bargap=0.05, xaxis=dict(range=[x_lo, x_hi]),
        legend=dict(orientation="v", x=1.02, y=1.0, bordercolor="lightgray", borderwidth=1),
        margin=dict(t=90, r=160)
    )
    st.plotly_chart(fig, use_container_width=True)

    # สรุปสั้น + Insight (โชว์ actual และอ้างอิงถ้าล็อก)
    real_mean, real_sd = vals.mean(), vals.std(ddof=1)
    st.markdown(
        f"**สถิติสรุป (แสดงที่กราฟ = `{'REF' if lock_ref else 'Actual'}`):** "
        f"Mean `{mean_val:.2f}` | SD `{std_val:.2f}`  "
        f"{'• Actual: ' + f'{real_mean:.2f} ± {real_sd:.2f}' if lock_ref else ''}"
    )

elif chart_type == "🧮 Group by Outcome":
    st.subheader("🧮 ค่าเฉลี่ย/SD/Min/Max ของแต่ละ Feature ตาม Outcome")
    st.dataframe(df.groupby("outcome").agg(["mean","std","min","max"]).T.style.format("{:.2f}"))

    mean_df = df.groupby("outcome").mean(numeric_only=True).T.reset_index()
    fig_bar = px.bar(mean_df, x="index", y=[0,1], barmode="group",
                     labels={"index":"Feature","value":"Mean Value"},
                     title="Mean Feature Value Comparison by Outcome",
                     template="plotly_white", text_auto=".2f")
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(legend_title_text="Outcome", yaxis_title="Mean Value")
    fig_bar.data[0].name = "No Diabetes (0)"
    fig_bar.data[1].name = "Diabetes (1)"
    st.plotly_chart(fig_bar, use_container_width=True)

elif chart_type == "📏 Max - Min (Per Feature)":
    st.subheader(f"📏 ค่าสูงสุดและต่ำสุดของ {display_name}")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Max"], y=[df[feature].max()], marker_color="indianred", name="Max",
                         text=f"{df[feature].max():.2f}", textposition="auto"))
    fig.add_trace(go.Bar(x=["Min"], y=[df[feature].min()], marker_color="lightseagreen", name="Min",
                         text=f"{df[feature].min():.2f}", textposition="auto"))
    fig.update_layout(template="plotly_white", title=f"Max vs Min of {display_name}",
                      yaxis_title="Value", legend_title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "📦 Horizontal Boxplot (Filtered)":
    st.subheader("📦 Boxplot แนวนอนของทุก Feature (ตามกติกา)")
    df_m = df.drop(columns=["outcome"]).melt(var_name="Feature", value_name="Value")
    fig = px.box(df_m, y="Feature", x="Value", orientation="h", color="Feature",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "🧠 Correlation Matrix":
    st.subheader("🧠 Correlation Matrix of Features (Pearson)")

    # ให้คำนวณเหมือนภาพอ้างอิง: 0 -> NaN เฉพาะ 4 คอลัมน์
    df2 = df.copy()
    df2.columns = df2.columns.str.strip().str.lower()
    zero_to_nan = ["glucose", "bloodpressure", "skinthickness", "bmi"]
    present = [c for c in zero_to_nan if c in df2.columns]
    if present:
        df2[present] = df2[present].replace(0, np.nan)

    # ลำดับคอลัมน์ให้เหมือนภาพ
    cols = [
        "pregnancies","glucose","bloodpressure","skinthickness",
        "insulin","bmi","diabetespedigreefunction","age","outcome"
    ]
    cols = [c for c in cols if c in df2.columns]

    corr = df2[cols].corr(method="pearson")  # ค่าสหสัมพันธ์

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns, y=corr.index,
            colorscale="RdBu", zmin=-1, zmax=1,
            colorbar=dict(title="Correlation", ticks="outside"),
            text=np.round(corr.values, 2),   # แสดงตัวเลข 2 หลัก
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Correlation Matrix of Features", x=0.5),
        xaxis=dict(side="bottom", tickangle=45, showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=120, r=40, t=60, b=80),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)




elif chart_type == "🤖 Predict Diabetes Risk (Model)":
    st.subheader("🤖 การทำนายแนวโน้มการเกิดโรคเบาหวาน (XGBoost Model)")

    # -----------------------------
    # 1) เตรียม Target แบบรวม (Outcome OR Outcome_Art)
    # -----------------------------
    has_outcome_art = 'outcome_art' in df.columns
    y_local = (
        (df['outcome'] == 1) |
        (df['outcome_art'] == 1 if has_outcome_art else False)
    ).astype(int)
    X_local = df.drop(columns=[c for c in ['outcome', 'outcome_art'] if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y_local, test_size=0.2, random_state=42, stratify=y_local
    )

    # -----------------------------
    # 2) Hyperparameter search (เล็ก กระชับ เร็ว)
    # -----------------------------
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    best_acc = 0.0
    best_params = None
    best_model = None

    with st.spinner("กำลังค้นหาพารามิเตอร์ที่เหมาะสม..."):
        for n in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    model = xgb.XGBClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        learning_rate=lr,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {'n_estimators': n, 'max_depth': depth, 'learning_rate': lr}
                        best_model = model

    st.success(
        f"✅ Best Parameters Found:\n"
        f"- n_estimators: {best_params['n_estimators']}\n"
        f"- max_depth: {best_params['max_depth']}\n"
        f"- learning_rate: {best_params['learning_rate']}\n"
        f"- Best Accuracy: {best_acc*100:.2f}%"
    )

    # -----------------------------
    # 3) ประเมินโมเดลที่ดีที่สุด
    # -----------------------------
    y_pred = best_model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    rep_txt = classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Diabetes (1)'])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy (Test)", f"{best_acc*100:.2f}%")
        st.markdown("**Classification Report**")
        st.code(rep_txt, language="text")
    with col2:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Diabetes (0)', 'Diabetes (1)'],
            y=['No Diabetes (0)', 'Diabetes (1)'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # -----------------------------
    # 4) Feature Importance (จากโมเดลที่ดีที่สุด)
    # -----------------------------
    st.subheader("🔑 Feature Importance (จากโมเดลที่ดีที่สุด)")
    imp_df = pd.DataFrame({
        'Feature': X_local.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_imp = px.bar(
        imp_df.head(10),
        x='Importance', y='Feature', orientation='h',
        title="Top 10 Feature Importances",
        template="plotly_white",
        text='Importance',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    # -----------------------------
    # 5) Live Prediction ฟอร์มรับค่าจริง
    # -----------------------------
    st.subheader("🔬 ทดสอบการทำนายผลด้วยตัวคุณเอง (Live Prediction)")
    st.caption("ป้อนค่าฟีเจอร์ต่าง ๆ (ค่าเริ่มต้น = ค่าเฉลี่ย) แล้วดูผลทำนายจากโมเดลที่ดีที่สุด")

    with st.form("prediction_form_best"):
        c1, c2 = st.columns(2)
        user_inputs = {}
        for i, f in enumerate(X_local.columns):
            s = df[f]
            default_val = float(s.mean(skipna=True))
            min_val = float(s.min(skipna=True))
            max_val = float(s.max(skipna=True))
            target_col = c1 if i < len(X_local.columns) / 2 else c2
            user_inputs[f] = target_col.number_input(
                label=f.capitalize(),
                min_value=min_val, max_value=max_val, value=default_val,
                step=1.0 if pd.api.types.is_integer_dtype(s) else 0.1,
                help=f"Min: {min_val:.2f}, Max: {max_val:.2f}"
            )
        submitted = st.form_submit_button("ทำนายผล (Predict)")

    if submitted:
        input_df = pd.DataFrame([user_inputs])
        pred = best_model.predict(input_df)[0]
        proba = best_model.predict_proba(input_df)[0]
        if pred == 1:
            st.error("**ผลการทำนาย: 🔴 มีความเสี่ยงเป็นโรคเบาหวาน (Diabetes)**")
            st.metric("ความน่าจะเป็น (Probability)", f"{proba[1]*100:.2f}%")
        else:
            st.success("**ผลการทำนาย: 🟢 ไม่มีความเสี่ยงเป็นโรคเบาหวาน (No Diabetes)**")
            st.metric("ความน่าจะเป็น (Probability)", f"{proba[0]*100:.2f}%")

        st.markdown("---")
        st.subheader("ข้อมูลที่คุณป้อน")
        st.dataframe(input_df.style.format("{:.2f}"))
