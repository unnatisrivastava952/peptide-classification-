import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# ----- Page Setup -----
st.set_page_config(page_title="Peptide Classification Web App", layout="wide")

# ----- Load model and scaler -----
try:
    logreg_model = joblib.load("logreg_model_30f.pkl")
    scaler = joblib.load("scaler_30f.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# ----- Constants -----
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CLASS_MAPPING = {
    0: "Antibacterial",
    1: "Antiviral",
    2: "Antimicrobial",
    3: "Antifungal"
}

# ----- Feature Extraction Functions -----
def extract_aac(sequence):
    seq_length = len(sequence)
    return np.array([sequence.count(aa) / seq_length for aa in AMINO_ACIDS])

def extract_physicochemical(sequence):
    hydro = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
             'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
    weight = {'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.1,'H':155.2,'I':131.2,'K':146.2,
              'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.2,'R':174.2,'S':105.1,'T':119.1,'V':117.1,
              'W':204.2,'Y':181.2}
    charge = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}
    instability = {'A':1.0,'C':1.2,'D':1.5,'E':1.3,'F':1.1,'G':0.7,'H':1.4,'I':1.3,'K':1.5,'L':1.2,
                   'M':1.2,'N':1.6,'P':1.3,'Q':1.2,'R':1.4,'S':1.0,'T':1.0,'V':1.2,'W':1.0,'Y':1.1}
    seq_length = len(sequence)
    avg_hydro = np.mean([hydro.get(aa, 0) for aa in sequence])
    avg_weight = np.mean([weight.get(aa, 0) for aa in sequence])
    avg_charge = sum([charge.get(aa, 0) for aa in sequence]) / seq_length
    aromaticity = sum([sequence.count(aa) for aa in "FWY"]) / seq_length
    instability_index = sum([instability.get(aa, 0) for aa in sequence]) / seq_length

    return np.array([
        avg_hydro, avg_weight, avg_charge, aromaticity, instability_index,
        seq_length,
        sequence.count("C") / seq_length,
        sequence.count("G") / seq_length,
        sequence.count("P") / seq_length,
        sequence.count("H") / seq_length,
    ])

def preprocess_sequence(sequence):
    sequence = sequence.upper()
    invalid_chars = set(sequence) - set(AMINO_ACIDS)
    if invalid_chars:
        raise ValueError(f"Invalid characters: {invalid_chars}")
    aac = extract_aac(sequence)
    physico = extract_physicochemical(sequence)
    features = np.hstack([aac, physico]).reshape(1, -1)
    return scaler.transform(features), aac, physico

# ----- Sidebar Navigation -----
page = st.sidebar.radio("Go to", ["🧬 Prediction", "📖 Manual", "👨‍🔬 Team"])
st.sidebar.markdown("---")

# ----- Manual Page -----
if page == "📖 Manual":
    st.title("📖 User Manual")
    st.markdown("""
    Welcome to the **Peptide Classification Web App**! This tool predicts the biological activity of peptides.

    ### Steps to Use:
    1. Enter a peptide sequence using standard amino acids.
    2. Click **Predict**.
    3. See classification and analysis of the peptide.

    ### Features:
    - Logistic Regression classification
    - Amino acid composition visualization
    - Physicochemical property breakdown

    For questions, contact the team.
    """)

elif page == "👨‍🔬 Team":
    st.markdown("<h1 style='color:#e91e63; text-align:center;'>👨‍🔬 Meet the Team</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);'>
            <h4 style='color:#2c3e50; margin-bottom:5px;'>Dr. Shailesh Kumar</h4>
            <p style='margin:0; color:gray;'>Staff Scientist IV</p>
            <ul style='font-size:16px; padding-left:20px;'>
                <li><b>Expertise:</b> Bioinformatics, Genomics, Big data analysis, Machine Learning (ML), Deep Learning, Artificial Intelligence (AI), and Plant Biotechnology
</li>
                <li><a href='https://www.nipgr.ac.in/research/dr_shailesh.php' target='_blank'>Profile Page</a></li>
                <li>Email: <a href='mailto:shailesh@nipgr.ac.in'>shailesh@nipgr.ac.in</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);'>
            <h4 style='color:#2c3e50; margin-bottom:5px;'>Unnati Srivastava</h4>
            <p style='margin:0; color:gray;'>Bioinformatics Student</p>
            <ul style='font-size:16px; padding-left:20px;'>
                <li>M.Sc. in Bioinformatics</li>
                <li>Project Lead for Peptide Classification</li>
                <li><a href='https://github.com/unnatisrivastava952/peptide-classification-' target='_blank'>GitHub Repo</a></li>
                <li>Email: <a href='mailto:srivastavaunnati93@gmail.com'>srivastavaunnati93@gmail.com</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)




    
                # ----- Prediction Page -----
elif page == "🧬 Prediction":
    # Header Section
    st.markdown("""
    <div style='text-align:center; padding: 30px 10px; background-color:#f0f8ff; border-radius:10px;'>
        <h1 style='color:#4a148c;'>🧬 Peptide Classification Web App</h1>
        <h3 style='color:#6a1b9a;'>Using <span style='color:#d81b60;'>Logistic Regression</span> to Classify Peptides</h3>
    </div>
    """, unsafe_allow_html=True)

    # Description Box
    st.markdown("""
    <div style='text-align: justify; font-size: 17px; background-color:#fef6e4; padding: 20px 30px; border-radius: 10px; border-left: 6px solid #ff9800;'>
        <p><b>Peptides</b> are short chains of amino acids linked by peptide bonds that play crucial roles in biological processes.</p>
        <p>This app uses a <b style='color:#e91e63;'>Logistic Regression model</b> to classify peptides into categories based on their amino acid composition and physicochemical properties.</p>
        <br>
        <b>✨ Key Features of Logistic Regression:</b>
        <ul>
            <li>Efficient and interpretable for multi-class tasks</li>
            <li>Calculates class membership probabilities</li>
            <li>Ideal for feature importance and linear decision boundaries</li>
            <li>Performs well on linearly separable data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Input Section
    st.markdown("### 🔍 <span style='color:#1976d2;'>Input Peptide Sequence</span>", unsafe_allow_html=True)
    if "peptide_sequence" not in st.session_state:
        st.session_state.peptide_sequence = ""

    peptide_sequence = st.text_input("Enter Sequence:", value=st.session_state.peptide_sequence)

    st.caption("Or select from example sequences:")
    example_sequences = [
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "GIGAVLNVAKKLLKSAKKLGQAAVAKAGKAAKKAAE",
        "GLFDIVKKVVGRGLL",
        "KKKKKKKKKKKKKKKK"
    ]
    col1, col2 = st.columns(2)
    for i, seq in enumerate(example_sequences):
        col = col1 if i % 2 == 0 else col2
        if col.button(seq, key=f"example_{i}"):
            st.session_state.peptide_sequence = seq
            st.experimental_rerun()

    # Prediction Action
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predict"):
        if not peptide_sequence:
            st.error("⚠️ Please enter a peptide sequence.")
        else:
            try:
                # Prediction Pipeline
                features, aac, physico = preprocess_sequence(peptide_sequence)
                prediction = logreg_model.predict(features)[0]
                predicted_class = CLASS_MAPPING.get(prediction, "Unknown")
                probs = logreg_model.predict_proba(features)[0]
                confidence = max(probs) * 100

                # Result Display Box
                st.markdown(f"""
                <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; border-left: 6px solid #4CAF50;'>
                    <h3 style='color:#2e7d32;'>✅ Prediction Result:</h3>
                    <p style='font-size:18px;'><b>Predicted Class:</b> <span style='color:#d81b60;'>{predicted_class}</span></p>
                    <p style='font-size:18px;'><b>Confidence Score:</b> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities Chart
                st.markdown("### 📊 <span style='color:#d81b60;'>Prediction Probabilities</span>", unsafe_allow_html=True)
                class_labels = [CLASS_MAPPING.get(c, f"Class {c}") for c in logreg_model.classes_]
                prob_df = pd.DataFrame({"Class": class_labels, "Probability": probs})
                fig = px.bar(prob_df, x="Class", y="Probability", color="Class", text=[f"{p:.2%}" for p in probs])
                fig.update_traces(textposition='auto')
                fig.update_layout(yaxis_range=[0, 1], title="Confidence per Class", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Sequence Statistics
                st.markdown("### 📌 <span style='color:#6a1b9a;'>Sequence Statistics</span>", unsafe_allow_html=True)
                stats_df = pd.DataFrame({
                    "Metric": ["Length", "Hydrophobicity", "Molecular Weight", "Charge", "Aromaticity"],
                    "Value": [
                        len(peptide_sequence),
                        f"{physico[0]:.2f}",
                        f"{physico[1]:.2f}",
                        f"{physico[2]:.2f}",
                        f"{physico[3]:.2%}"
                    ]
                })
                st.table(stats_df)

                # AAC Chart
                st.markdown("### 🧪 <span style='color:#0097a7;'>Amino Acid Composition (AAC)</span>", unsafe_allow_html=True)
                aac_df = pd.DataFrame({"Amino Acid": list(AMINO_ACIDS), "Percentage": aac})
                fig2 = px.bar(aac_df, x="Amino Acid", y="Percentage", color="Amino Acid", text=[f"{p:.2%}" for p in aac])
                fig2.update_traces(textposition='auto')
                fig2.update_layout(title="AAC Bar Chart", yaxis_range=[0, 1], template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"❗ Unexpected error: {e}")

# ----- Footer -----
st.markdown("<hr><center><small style='color:gray;'> Built with using <b>Streamlit</b> and <b>Plotly</b> | © 2025 Peptide Classifier</small></center>", unsafe_allow_html=True)
