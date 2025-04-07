
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
    if "run_prediction" not in st.session_state:
        st.session_state.run_prediction = False

    peptide_sequence = st.text_input("Enter Sequence:", value=st.session_state.peptide_sequence)

    st.caption(" Select from example sequences:")
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
            st.session_state.run_prediction = True  # Safe rerun flag

    # Safe rerun trigger
    if st.session_state.run_prediction:
        st.session_state.run_prediction = False
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
