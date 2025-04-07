import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

# ----- Load trained model and scaler -----
try:
    logreg_model = joblib.load("logreg_model_30f.pkl")
    scaler = joblib.load("scaler_30f.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ----- Amino acid and class definitions -----
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
    if seq_length == 0:
        raise ValueError("Sequence cannot be empty")
    aac_features = [sequence.count(aa) / seq_length for aa in AMINO_ACIDS]
    return np.array(aac_features)

def extract_physicochemical(sequence):
    hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
                      'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
                      'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                      'W': -0.9, 'Y': -1.3}
    molecular_weight = {'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2, 'G': 75.1,
                        'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2, 'M': 149.2, 'N': 132.1,
                        'P': 115.1, 'Q': 146.2, 'R': 174.2, 'S': 105.1, 'T': 119.1, 'V': 117.1,
                        'W': 204.2, 'Y': 181.2}
    seq_length = len(sequence)
    if seq_length == 0:
        raise ValueError("Sequence cannot be empty")

    avg_hydro = np.mean([hydrophobicity.get(aa, 0) for aa in sequence])
    avg_weight = np.mean([molecular_weight.get(aa, 0) for aa in sequence])
    avg_charge = sum([{'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}.get(aa, 0) for aa in sequence]) / seq_length
    aromaticity = sum([sequence.count(aa) for aa in "FWY"]) / seq_length
    instability_index = sum([{'A': 1.0, 'C': 1.2, 'D': 1.5, 'E': 1.3, 'F': 1.1, 'G': 0.7,
                               'H': 1.4, 'I': 1.3, 'K': 1.5, 'L': 1.2, 'M': 1.2, 'N': 1.6,
                               'P': 1.3, 'Q': 1.2, 'R': 1.4, 'S': 1.0, 'T': 1.0, 'V': 1.2,
                               'W': 1.0, 'Y': 1.1}.get(aa, 0) for aa in sequence]) / seq_length
    return np.array([
        avg_hydro, avg_weight, avg_charge,
        aromaticity, instability_index,
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
        raise ValueError(f"Invalid amino acids found: {invalid_chars}")
    aac = extract_aac(sequence)
    physico = extract_physicochemical(sequence)
    features = np.hstack([aac, physico]).reshape(1, -1)
    features = scaler.transform(features)
    return features, aac, physico

# ----- UI Layout -----
st.set_page_config(page_title="Peptide Classification Web App", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["🧬 Prediction", "📖 Manual"])
st.sidebar.link_button("Team Page", "https://nipgr.ac.in/research/dr_shailesh.php")

# Custom styles
st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-left: 5%; padding-right: 5%; }
        h1, h3 { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ----- Manual Page -----
if page == "📖 Manual":
    st.title("📖 User Manual")
    st.markdown("""
    Welcome to the **Peptide Classification Web App**! This manual provides step-by-step instructions on how to use the application.

    ### 📌 How to Use the App
    1. **Enter a Peptide Sequence**  
       - Use only standard amino acids (A, C, D, E, etc.).
    2. **Click "Predict"**  
       - The model analyzes amino acid composition and physicochemical properties.
    3. **View Results**  
       - Get prediction, confidence, and graphs.
    4. **Explore Features**  
       - See hydrophobicity, molecular weight, etc.

    ### 📊 Features
    - **Machine Learning (Logistic Regression)**
    - **Amino Acid Composition Charts**
    - **Prediction Probabilities**

    ### ❓ Need Help?
    Visit the **Team Page** linked in the sidebar.
    """)

# ----- Prediction Page -----
elif page == "🧬 Prediction":
    st.markdown("<h1>🧬 Peptide Classification Web App</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Using Logistic Regression to Classify Peptides</h3>", unsafe_allow_html=True)
    st.markdown("""
**Peptides** are short chains of amino acids linked by peptide bonds, playing crucial roles in biological processes.  
This app uses a **Logistic Regression model** to classify peptides into categories based on their amino acid composition  
and physicochemical properties. Enter a peptide sequence to predict its potential biological activity.
""")

    st.markdown("""
    <style>
        .main .block-container {
            max-width: 100%;
            padding-left: 5%;
            padding-right: 5%;
            padding-top: 2rem;
        }
        h1, h3 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


    

   st.subheader("Input Peptide Sequence")
    peptide_sequence = st.text_input(
        "Enter Sequence (only A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y):", ""
    )

    if st.button("Predict"):
        if not peptide_sequence:
            st.error("⚠️ Please enter a peptide sequence.")
        else:
            try:
                input_features, aac, physico = preprocess_sequence(peptide_sequence)
                prediction = logreg_model.predict(input_features)[0]
                predicted_class = CLASS_MAPPING.get(prediction, "Unknown")
                probs = logreg_model.predict_proba(input_features)[0]
                confidence = max(probs) * 100

                st.success(f"🧪 Predicted Class: {predicted_class}")
                st.write(f"Confidence Score: {confidence:.2f}%")

                model_classes = logreg_model.classes_
                class_labels = [CLASS_MAPPING.get(c, f"Class {c}") for c in model_classes]

                prob_df = pd.DataFrame({
                    "Class": class_labels,
                    "Probability": probs
                })
                fig_prob = px.bar(prob_df, x="Class", y="Probability", color="Class", range_y=[0, 1],
                                  title="Prediction Probabilities", text=[f"{p:.2%}" for p in probs])
                fig_prob.update_traces(textposition='auto')
                st.plotly_chart(fig_prob, use_container_width=True)

                # Display physicochemical stats
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

                # AAC Bar Chart
                st.subheader("Amino Acid Composition")
                aac_df = pd.DataFrame({
                    "Amino Acid": list(AMINO_ACIDS),
                    "Percentage": aac
                })
                fig_aac = px.bar(aac_df, x="Amino Acid", y="Percentage", color="Amino Acid",
                                 title="Amino Acid Composition", range_y=[0, 1],
                                 text=[f"{p:.2%}" for p in aac])
                fig_aac.update_traces(textposition='auto')
                st.plotly_chart(fig_aac, use_container_width=True)

            except ValueError as e:
                st.error(f"⚠️ Error: {str(e)}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {str(e)}")

# Footer
st.markdown("<hr><center><small>Built with using Streamlit and Plotly</small></center>", unsafe_allow_html=True)
