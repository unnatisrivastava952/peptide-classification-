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

# ----- Team Page -----
elif page == "👨‍🔬 Team":
    st.title("👨‍🔬 Meet the Team")
    cols = st.columns(2)

    with cols[0]:
        try:
            img1 = Image.open("team1.jpg")
            st.image(img1, width=150)
        except:
            st.warning("team1.jpg not found")
        st.subheader("Dr. Shailesh Kumar")
        st.caption("Staff Scientist IV")

    with cols[1]:
        try:
            img2 = Image.open("/home/namrata/Downloads/IMG_20250407_150441.jpg")
            st.image(img2, width=150)
        except:
            st.warning("/home/namrata/Downloads/IMG_20250407_150441.jpg")
        st.subheader("Unnati Srivastava")
        st.caption("Bioinformatics Student")

    

# ----- Prediction Page -----
elif page == "🧬 Prediction":
    st.markdown("<h1 style='text-align:center;'>🧬 Peptide Classification Web App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Using Logistic Regression to Classify Peptides</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: justify; font-size: 18px; padding: 10px 30px;'>
    Peptides are short chains of amino acids linked by peptide bonds, playing crucial roles in biological processes.  
    This app uses a <b>Logistic Regression model</b> to classify peptides into categories based on their amino acid composition  
    and physicochemical properties. Enter a peptide sequence to predict its potential biological activity.

    <br><br>
    <b> Key Features of Logistic Regression:</b>
    <ul>
      <li>Efficient and interpretable for binary and multi-class classification tasks</li>
      <li>Calculates probabilities of class membership</li>
      <li>Useful when feature importance and decision boundaries are needed</li>
      <li>Less prone to overfitting with fewer parameters</li>
      <li>Performs well on linearly separable data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


    st.subheader("Input Peptide Sequence")
    if "peptide_sequence" not in st.session_state:
        st.session_state.peptide_sequence = ""

    peptide_sequence = st.text_input("Enter Sequence:", value=st.session_state.peptide_sequence)

    st.caption("Or select an example sequence:")
    example_sequences = [
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "GIGAVLNVAKKLLKSAKKLGQAAVAKAGKAAKKAAE",
        "GLFDIVKKVVGRGLL",
        "KKKKKKKKKKKKKKKK"
    ]
    col1, col2 = st.columns(2)
    for i, seq in enumerate(example_sequences):
        if i % 2 == 0:
            if col1.button(seq, key=f"example_{i}"):
                st.session_state.peptide_sequence = seq
                st.experimental_rerun()
        else:
            if col2.button(seq, key=f"example_{i}"):
                st.session_state.peptide_sequence = seq
                st.experimental_rerun()

    if st.button("Predict"):
        if not peptide_sequence:
            st.error("⚠️ Please enter a peptide sequence.")
        else:
            try:
                features, aac, physico = preprocess_sequence(peptide_sequence)
                prediction = logreg_model.predict(features)[0]
                predicted_class = CLASS_MAPPING.get(prediction, "Unknown")
                probs = logreg_model.predict_proba(features)[0]
                confidence = max(probs) * 100

                st.success(f"🧪 Predicted Class: {predicted_class}")
                st.write(f"Confidence Score: **{confidence:.2f}%**")

                # Probability Chart
                class_labels = [CLASS_MAPPING.get(c, f"Class {c}") for c in logreg_model.classes_]
                prob_df = pd.DataFrame({"Class": class_labels, "Probability": probs})
                fig = px.bar(prob_df, x="Class", y="Probability", color="Class", text=[f"{p:.2%}" for p in probs])
                fig.update_traces(textposition='auto')
                fig.update_layout(title="Prediction Probabilities", yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

                # Sequence Statistics
                st.subheader("Sequence Statistics")
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
                st.subheader("Amino Acid Composition")
                aac_df = pd.DataFrame({"Amino Acid": list(AMINO_ACIDS), "Percentage": aac})
                fig2 = px.bar(aac_df, x="Amino Acid", y="Percentage", color="Amino Acid", text=[f"{p:.2%}" for p in aac])
                fig2.update_traces(textposition='auto')
                fig2.update_layout(title="AAC", yaxis_range=[0, 1])
                st.plotly_chart(fig2, use_container_width=True)

            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")

# ----- Footer -----
st.markdown("<hr><center><small>Built with using Streamlit and Plotly</small></center>", unsafe_allow_html=True)
