# ───────────── app.py  (PART 1 / 2) ─────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fpdf import FPDF
import tempfile, os, smtplib
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="HR Analytics Dashboard", page_icon="📊")

# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.title("📘 How to Use")
    st.markdown("""
1. **Upload** HR CSV  
2. **Confirm/adjust** column mapping  
3. Browse **Overview / Visuals**  
4. Run **Predictions**  
5. **Generate & Email** PDF report
""")
    st.info("Built by Yaswanth Kumar\n📧 yaswanth0908@gmail.com")

st.title("📊 HR Analytics Dashboard")
uploaded_file = st.file_uploader("📥 Upload your HR CSV", type=["csv"])

# ── Column‑mapping helpers ───────────────────────
DEFAULT_MAP = {
    "Attrition":       ["attrition", "left_company"],
    "Department":      ["department", "dept"],
    "JobSatisfaction": ["job_satisfaction", "satisfaction"],
    "YearsAtCompany":  ["years_at_company", "tenure"],
    "MonthlyIncome":   ["monthly_income", "salary"],
    "Gender":          ["gender", "sex"],
    "OverTime":        ["overtime", "ot"],
    "Age":             ["age"],
    "JobLevel":        ["job_level", "level"],
}

def auto_map(df):
    ren = {}
    for std, aliases in DEFAULT_MAP.items():
        for col in df.columns:
            if col.lower().replace(" ", "_") in aliases:
                ren[col] = std
    return df.rename(columns=ren)

def manual_map(df):
    prompts = {
        "Attrition":       "Attrition (Yes/No):",
        "Department":      "Department:",           "JobSatisfaction":"Job Satisfaction:",
        "YearsAtCompany":  "Years at Company:",     "MonthlyIncome":  "Monthly Salary:",
        "Gender":          "Gender:",               "OverTime":       "OverTime (Yes/No):",
        "Age":             "Age:",                  "JobLevel":       "Job Level:",
    }
    for std, q in prompts.items():
        if std not in df.columns:
            sel = st.selectbox(q, df.columns, key=f"map_{std}")
            df = df.rename(columns={sel: std})
    return df

# ── Main Pipeline ────────────────────────────────
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = auto_map(df)
    df = manual_map(df)
    st.success("✅ File mapped!")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Overview", "📊 Visuals", "🧠 Prediction", "📄 Reports"]
    )

    # ── OVERVIEW TAB ─────────────────────────────
    with tab1:
        st.header("📈 Summary Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Employees", df.shape[0])
        attr = df[df["Attrition"].str.lower()=="yes"].shape[0] / df.shape[0]
        c2.metric("Attrition Rate", f"{attr*100:.2f}%")
        c3.metric("Avg Satisfaction", round(df["JobSatisfaction"].mean(),2))
        c4.metric("Avg Tenure", round(df["YearsAtCompany"].mean(),2))
        st.dataframe(df.head())

    # ── VISUALS TAB ─────────────────────────────
    with tab2:
        st.header("📊 Charts & Visuals")
        if {"Attrition","Department"}.issubset(df.columns):
            st.subheader("Attrition by Department")
            st.bar_chart(df[df["Attrition"].str.lower()=="yes"]["Department"].value_counts())

        if "Gender" in df.columns:
            st.subheader("Gender Distribution")
            g = df["Gender"].value_counts().reset_index()
            g.columns=["Gender","Count"]
            st.plotly_chart(px.pie(g, names="Gender", values="Count"))

        if {"Attrition","JobSatisfaction"}.issubset(df.columns):
            st.subheader("Job Satisfaction vs Attrition")
            st.plotly_chart(px.box(df, x="Attrition", y="JobSatisfaction", color="Attrition"))

        if {"MonthlyIncome","Department"}.issubset(df.columns):
            st.subheader("Average Income by Department")
            inc = df.groupby("Department")["MonthlyIncome"].mean().reset_index()
            st.plotly_chart(px.bar(inc, x="Department", y="MonthlyIncome"))

        st.subheader("Correlation Heatmap")
        num = df.select_dtypes("number")
        if not num.empty:
            fig, ax = plt.subplots(figsize=(10,5))
            sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
# ───────────── app.py  (PART 2 / 2) ─────────────
    # ── PREDICTION TAB ───────────────────────────
    with tab3:
        st.header("🧠 Predict Employee Attrition")
        model_type = st.radio("Model", ["Logistic Regression", "Random Forest"])
        work = df.dropna().copy()
        work["Attrition"] = work["Attrition"].apply(lambda x: 1 if str(x).lower()=="yes" else 0)
        features = ["Age","MonthlyIncome","JobLevel","JobSatisfaction","YearsAtCompany","OverTime"]
        work = work[features+["Attrition"]]
        for col in work.select_dtypes("object"):
            work[col] = LabelEncoder().fit_transform(work[col])

        X, y = work.drop("Attrition", axis=1), work["Attrition"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)

        model = (LogisticRegression(max_iter=1000) if model_type=="Logistic Regression"
                 else RandomForestClassifier())
        model.fit(X_tr,y_tr)
        st.success(f"Accuracy: {accuracy_score(y_te, model.predict(X_te))*100:.2f}%")

        df_pred = df.copy()
        for col in features:
            if df_pred[col].dtype=="object":
                df_pred[col] = LabelEncoder().fit_transform(df_pred[col].astype(str))
        df_pred["Predicted Attrition"] = model.predict(df_pred[features])
        df_pred["Attrition Probability (%)"] = model.predict_proba(df_pred[features])[:,1]*100
        st.dataframe(df_pred[features+['Predicted Attrition','Attrition Probability (%)']].head(20))
        st.download_button("📥 Download Predictions CSV",
                           df_pred.to_csv(index=False).encode(),
                           "attrition_predictions.csv")

    # ── REPORT TAB (PDF + EMAIL) ─────────────────
    with tab4:
        st.header("📄 Generate PDF Report")

        def safe(t:str):
            return (t.replace("–","-").replace("—","-")
                     .replace("‑","-").replace("“","\"").replace("”","\""))

        def pdf_report(data):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
            pdf.cell(0,10,safe("HR Analytics Report"),ln=True,align="C")
            pdf.ln(5)
            pdf.cell(0,10,f"Total Employees: {data.shape[0]}",ln=True)
            pdf.cell(0,10,f"Avg Tenure: {data['YearsAtCompany'].mean():.2f}",ln=True)
            rate = data[data['Attrition'].str.lower()=='yes'].shape[0]/data.shape[0]*100
            pdf.cell(0,10,f"Attrition Rate: {rate:.2f}%",ln=True)
            return pdf

        def send_email(recipient,path):
            msg = EmailMessage()
            msg['Subject'] = "HR Analytics Report"
            msg['From'] = "yaswanth0809@gmail.com"
            msg['To'] = recipient
            msg.set_content("Attached is your HR report.")
            with open(path,"rb") as f:
                msg.add_attachment(f.read(),maintype="application",subtype="pdf",
                                   filename=os.path.basename(path))
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
                s.login("yaswanth0809@gmail.com","bkvn knna doua hait")
                s.send_message(msg)

        if st.button("📄 Generate PDF"):
            pdf = pdf_report(df)
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                pdf.output(tmp.name)
                st.session_state["pdf_path"] = tmp.name
                st.success("✅ PDF generated!")

        pdf_path = st.session_state.get("pdf_path")
        if pdf_path:
            with open(pdf_path,"rb") as f:
                st.download_button("⬇️ Download Report",f,"HR_Report.pdf","application/pdf")

            with st.expander("📧 Email This Report"):
                rcpt = st.text_input("Recipient Email")
                if st.button("Send Email"):
                    if rcpt:
                        try:
                            send_email(rcpt,pdf_path)
                            st.success(f"✅ Email sent to {rcpt}")
                        except Exception as e:
                            st.error(f"❌ Email error: {e}")
                    else:
                        st.warning("⚠️ Enter recipient email.")

# ───────────── END OF FILE ─────────────
