import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import groq
from groq import Groq

# Initialize Groq client (replace with your actual API key)
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# Load the model (assuming it's already trained and saved as 'model_1.h5')
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_1.h5')

model = load_model()

# List of all possible elements for A and B sites
elements_a = [
    'Ac', 'Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 
    'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 
    'Hg', 'Ho', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 
    'Nb', 'Nd', 'Ni', 'Np', 'Os', 'Pa', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 
    'Rb', 'Re', 'Rh', 'Ru', 'Sb', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 
    'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'
]
elements_b = elements_a.copy()  # Same list for B site

# Normalize function
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

# Prepare feature vector function
def prepare_feature_vector(input_params):
    """
    Prepare feature vector exactly matching the 170-feature dataset structure
    """
    # Numeric features
    numeric_features = [
        'r(AXII)(Å)', 'r(AVI)(Å)', 'r(BVI)(Å)', 
        'EN(A)', 'EN(B)', 
        'l(A-O)(Å)', 'l(B-O)(Å)', 
        'ΔENR'
    ]

    normalization_ranges = {
        'r(AXII)(Å)': (0.5, 2.0),
        'r(AVI)(Å)': (0.5, 2.0),
        'r(BVI)(Å)': (0.5, 2.0),
        'EN(A)': (1.0, 4.0),
        'EN(B)': (1.0, 4.0),
        'l(A-O)(Å)': (1.0, 3.0),
        'l(B-O)(Å)': (1.0, 3.0),
        'ΔENR': (0.0, 3.0)
    }

    features = []

    # 1. Normalize numeric features
    for param in numeric_features:
        min_val, max_val = normalization_ranges[param]
        features.append(normalize(input_params[param], min_val, max_val))

    # 2. Additional numeric features
    features.append(input_params.get('tG', 0))
    features.append(input_params.get('μ', 0))

    # 3. Valency One-Hot Encoding (12 features total)
    valency_options = ['-', '0', '1', '2', '3', '4', '5']

    # A site valency (6 features)
    features.extend([1 if str(input_params['v(A)']) == val else 0 for val in valency_options])
    
    # B site valency (6 features)
    features.extend([1 if str(input_params['v(B)']) == val else 0 for val in valency_options])

    # 4. Element One-Hot Encoding for A site (76 features)
    features.extend([1 if input_params['First Element'] == elem else 0 for elem in elements_a])

    # 5. Element One-Hot Encoding for B site (76 features)
    features.extend([1 if input_params['Second Element'] == elem else 0 for elem in elements_b])

    # Ensure exactly 170 features
    if len(features) > 170:
        features = features[:170]
    elif len(features) < 170:
        features.extend([0] * (170 - len(features)))

    return np.array(features, dtype=np.float32)

# Predict Crystal Structure
def predict_crystal_structure(input_params):
    """
    Predict crystal structure using prepared features
    """
    try:
        # Prepare features
        features = prepare_feature_vector(input_params)
        features = features.reshape(1, -1)  # Reshape for prediction

        # Predict
        prediction = model.predict(features)
        
        # Interpret prediction
        structure_classes = ['cubic', 'orthorhombic', 'rhombohedral', 'tetragonal']
        predicted_structure = structure_classes[np.argmax(prediction)]
        
        return {
            'predicted_structure': predicted_structure,
            'stability_index': float(np.max(prediction))
        }
    
    except Exception as e:
        st.error(f"Detailed Prediction Error: {e}")
        return None
    
def get_ai_explanation(prediction_result, input_params):
    """
    Use Groq API to generate an explanation of the crystal structure
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a materials science expert specializing in crystal structure prediction."
                },
                {
                    "role": "user",
                    "content": f"""Provide a scientific explanation for the predicted crystal structure 
                    based on these parameters:
                    Predicted Structure: {prediction_result['predicted_structure']}
                    Stability Index: {prediction_result['stability_index']:.4f}
                    
                    Input Parameters:
                    {', '.join(f'{k}: {v}' for k, v in input_params.items())}
                    
                    Explain the structural characteristics, potential bonding mechanisms, 
                    and factors influencing the crystal structure in a concise, technical manner."""
                }
            ],
            model="llama3-70b-8192",
            max_tokens=300
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# Streamlit UI
def main():
    st.title("Crystal Structure Prediction")

    # Input fields for all required parameters
    input_params = {}

    input_params['r(AXII)(Å)'] = st.number_input("r(AXII)(Å)", value=1.12)
    input_params['r(AVI)(Å)'] = st.number_input("r(AVI)(Å)", value=1.12)
    input_params['r(BVI)(Å)'] = st.number_input("r(BVI)(Å)", value=1.12)
    input_params['EN(A)'] = st.number_input("EN(A)", value=1.10)
    input_params['EN(B)'] = st.number_input("EN(B)", value=1.10)
    input_params['l(A-O)(Å)'] = st.number_input("l(A-O)(Å)", value=0.0)
    input_params['l(B-O)(Å)'] = st.number_input("l(B-O)(Å)", value=0.0)
    input_params['ΔENR'] = st.number_input("ΔENR", value=-3.248)
    input_params['tG'] = st.number_input("tG", value=0.707107)
    input_params['μ'] = st.number_input("μ", value=0.800000)
    
    # Valency options (A and B site)
    valency_a = st.selectbox('Valency for A site', ['-', '0', '1', '2', '3', '4', '5'])
    input_params['v(A)'] = valency_a

    valency_b = st.selectbox('Valency for B site', ['-', '0', '1', '2', '3', '4', '5'])
    input_params['v(B)'] = valency_b

    # Element selection (A and B sites)
    element_a = st.selectbox('Element for A site', elements_a)
    input_params['First Element'] = element_a

    element_b = st.selectbox('Element for B site', elements_b)
    input_params['Second Element'] = element_b

    # Predict button
    if st.button("Predict Crystal Structure"):
        result = predict_crystal_structure(input_params)
        
        if result:
            st.write(f"Predicted Structure: {result['predicted_structure']}")
            st.write(f"Stability Index: {result['stability_index']:.2f}")

            explanation = get_ai_explanation(result, input_params)
    
            # Display Explanation
            st.subheader("Scientific Explanation")
            st.info(explanation)

# Run the app
if __name__ == "__main__":
    main()
