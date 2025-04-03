import streamlit as st

def show_manual():
    st.title("📖 User Manual")
    st.markdown("""
    Welcome to the **Peptide Classification Web App**! This manual provides step-by-step instructions on how to use the application.

    ### 📌 How to Use the App
    1. **Enter a Peptide Sequence**  
       - Input a peptide sequence consisting of the standard 20 amino acids (A, C, D, E, etc.).
    2. **Click "Predict"**  
       - The app will analyze the sequence using **Amino Acid Composition (AAC)** and **Physicochemical Properties**.
    3. **View Results**  
       - The app will predict whether the sequence is *Antibacterial, Antiviral, Antimicrobial,* or *Antifungal*.
       - Confidence scores and probability distributions will be displayed.
    4. **Analyze Features**  
       - Additional statistics like amino acid composition and physicochemical properties are provided.

    ### 📊 Features
    - **ML-based classification** using Logistic Regression.
    - **Visualization of Amino Acid Composition**.
    - **Confidence Scores** for better decision-making.

    ### ❓ Need Help?
    If you encounter any issues, feel free to contact the development team via the **Team Page**.

    ---
    """)

if __name__ == "__main__":
    show_manual()
