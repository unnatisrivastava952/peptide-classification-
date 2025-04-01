import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained Logistic Regression model and scaler
try:
    logreg_model = joblib.load("logreg_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Prediction class mapping
CLASS_MAPPING = {
    0: "Antibacterial",
    1: "Antiviral",
    2: "Antimicrobial",
    3: "Antifungal"
}

def extract_aac(sequence):
    """Calculate AAC (Amino Acid Composition) features."""
    seq_length = len(sequence)
    if seq_length == 0:
        raise ValueError("Sequence cannot be empty")
    aac_features = [sequence.count(aa) / seq_length for aa in AMINO_ACIDS]
    return np.array(aac_features)

def extract_physicochemical(sequence):
    """Extract 10 physicochemical properties."""
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
       
    avg_hydrophobicity = np.mean([hydrophobicity.get(aa, 0) for aa in sequence])
    avg_molecular_weight = np.mean([molecular_weight.get(aa, 0) for aa in sequence])
    avg_charge = sum([{'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}.get(aa, 0) for aa in sequence]) / seq_length
    aromaticity = sum([sequence.count(aa) for aa in "FWY"]) / seq_length
    instability_index = sum([{'A': 1.0, 'C': 1.2, 'D': 1.5, 'E': 1.3, 'F': 1.1, 'G': 0.7,
                            'H': 1.4, 'I': 1.3, 'K': 1.5, 'L': 1.2, 'M': 1.2, 'N': 1.6,
                            'P': 1.3, 'Q': 1.2, 'R': 1.4, 'S': 1.0, 'T': 1.0, 'V': 1.2,
                            'W': 1.0, 'Y': 1.1}.get(aa, 0) for aa in sequence]) / seq_length
   
    return np.array([
        avg_hydrophobicity, avg_molecular_weight, avg_charge,
        aromaticity, instability_index,
        seq_length,
        sequence.count("C") / seq_length,
        sequence.count("G") / seq_length,
        sequence.count("P") / seq_length,
        sequence.count("H") / seq_length,
    ])

def preprocess_sequence(sequence):
    """Extract 30 features and standardize them using the pre-trained scaler."""
    sequence = sequence.upper()
    invalid_chars = set(sequence) - set(AMINO_ACIDS)
    if invalid_chars:
        raise ValueError(f"Invalid amino acids found: {invalid_chars}")
   
    aac = extract_aac(sequence)
    physico = extract_physicochemical(sequence)
   
    features = np.hstack([aac, physico]).reshape(1, -1)
    features = scaler.transform(features)
    return features, aac, physico

# Set full-width layout
st.set_page_config(page_title="Peptide Classification Web App", layout="wide")

# Custom CSS for full-width content
st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 100%;
            padding-left: 5%;
            padding-right: 5%;
        }
       
        h1 {
            text-align: center;
            color: #212529;
            font-size: 36px;
            font-weight: bold;
        }

        h3 {
            text-align: center;
            color: blue;
            font-size: 22px;
            font-weight: bold;
        }

        p {
            text-align: justify;
            font-size: 16px;
            color: darkgreen;
        }

        .stButton>button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1>🧬 Peptide Classification Web App</h1>", unsafe_allow_html=True)

# Description of the model
st.markdown("""
<p>
<b>Logistic Regression</b> is a statistical model used for classification tasks. It estimates probabilities using a logistic function, which outputs values between 0 and 1, representing the likelihood of a certain class.
In our peptide classification task, the model predicts whether a given peptide belongs to one of the following categories:
<b>Antibacterial</b>, <b>Antiviral</b>, <b>Antimicrobial</b>, or <b>Antifungal</b>.
</p>
""", unsafe_allow_html=True)

# Subtitle
st.markdown("<h3>Using Logistic Regression to classify peptides</h3>", unsafe_allow_html=True)

# Styled input field for peptide sequence
st.markdown("<h5 style='color:red; font-weight:bold;'>Enter Peptide Sequence:</h5>", unsafe_allow_html=True)
peptide_sequence = st.text_input(
    "Enter Peptide Sequence (using A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y):",
    ""
)

# Example sequences
st.markdown("""
<p><b>Example Sequences:</b></p>
<ul>
    <li>ACDEFGHIKLMNPQRSTVWY</li>
    <li>FVGQKIEKFTGQWEFD</li>
    <li>MAEQGRFHLWIVKSP</li>
</ul>
<p>Copy and paste any of these sequences into the input box to try the prediction.</p>
""", unsafe_allow_html=True)

# Prediction output description
st.markdown("""
<p><b>Output Description:</b></p>
<p>After you submit the peptide sequence, the app will predict the category to which your peptide belongs. The model will display the class (Antibacterial, Antiviral, Antimicrobial, or Antifungal) with a confidence score. Additionally, it will show the prediction probabilities for each class and a table with the sequence statistics and amino acid composition.</p>
""", unsafe_allow_html=True)

# Button and prediction logic
if st.button("Predict"):
    if not peptide_sequence:
        st.markdown(
            "<p style='color:red; font-weight:bold; font-size:16px;'>⚠️ Please enter a peptide sequence.</p>",
            unsafe_allow_html=True
        )
    else:
        try:
            input_features, aac_features, physico_features = preprocess_sequence(peptide_sequence)

            # Predict using the Logistic Regression model
            prediction = logreg_model.predict(input_features)[0]
            predicted_class = CLASS_MAPPING.get(prediction, "Unknown")

            st.success(f"🧪 Predicted Class: {predicted_class}")

            if hasattr(logreg_model, "predict_proba"):
                probs = logreg_model.predict_proba(input_features)[0]
                confidence = max(probs) * 100
                st.write(f"Confidence Score: {confidence:.2f}%")
                st.write("Prediction Probabilities:")
                for class_idx, prob in enumerate(probs):
                    st.write(f"{CLASS_MAPPING[class_idx]}: {prob:.2%}")

                # Plot prediction probabilities
                fig, ax = plt.subplots()
                ax.bar(CLASS_MAPPING.values(), probs, color='skyblue')
                ax.set_ylim(0, 1)
                ax.set_title("Prediction Probabilities")
                ax.set_ylabel("Probability")
                st.pyplot(fig)

            # Display sequence statistics and AAC in HTML Table
            st.subheader("Sequence Statistics and Amino Acid Composition")
           
            stats_data = {
                "Length": len(peptide_sequence),
                "Hydrophobicity": f"{physico_features[0]:.2f}",
                "Mol. Weight": f"{physico_features[1]:.2f}",
                "Charge": f"{physico_features[2]:.2f}",
                "Aromaticity": f"{physico_features[3]:.2%}"
            }
            aac_dict = {aa: f"{count:.2%}" for aa, count in zip(AMINO_ACIDS, aac_features)}
           
            combined_data = {**stats_data, **aac_dict}
           
            table_html = """
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {background-color: #f9f9f9;}
                tr:hover {background-color: #f5f5f5;}
            </style>
            <table>
                <tr>
            """
            for key in combined_data.keys():
                table_html += f"<th>{key}</th>"
            table_html += "</tr><tr>"
           
            for value in combined_data.values():
                table_html += f"<td>{value}</td>"
            table_html += "</tr></table>"
           
            st.markdown(table_html, unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"⚠️ Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")

# Create a help link or section for user guidance
st.markdown("""
<p><b>Need help?</b> <a href="https://your-website-link-to-help-docs" target="_blank">Click here</a> to access the user manual and learn how to use this app effectively.</p>
""", unsafe_allow_html=True)

