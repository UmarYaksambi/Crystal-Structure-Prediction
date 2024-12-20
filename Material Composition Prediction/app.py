import streamlit as st
import groq
import json
from typing import Dict, Any

# Configure the page
st.set_page_config(
    page_title="Material Composition Predictor",
    layout="wide"
)

# Initialize Groq client using Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = groq.Client(api_key=groq_api_key)
except Exception as e:
    st.error("Error loading GROQ API key from secrets. Please ensure your .streamlit/secrets.toml file is properly configured.")
    st.stop()

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the LLM response text and convert it to the expected format
    """
    try:
        # First try direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from the text
        try:
            # Look for text between curly braces
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except Exception:
            pass
        
        # If JSON extraction fails, try to parse structured text
        try:
            result = {
                "matrix": {"material": "", "proportion": 0.0},
                "reinforcement": {"material": "", "proportion": 0.0},
                "additive": {"material": "", "proportion": 0.0}
            }
            
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip().lower()
                if 'matrix' in line:
                    parts = line.split(':')[-1].split(',')
                    for part in parts:
                        if '%' in part:
                            result['matrix']['proportion'] = float(part.replace('%', '').strip())
                        else:
                            result['matrix']['material'] = part.strip()
                elif 'reinforcement' in line:
                    parts = line.split(':')[-1].split(',')
                    for part in parts:
                        if '%' in part:
                            result['reinforcement']['proportion'] = float(part.replace('%', '').strip())
                        else:
                            result['reinforcement']['material'] = part.strip()
                elif 'additive' in line:
                    parts = line.split(':')[-1].split(',')
                    for part in parts:
                        if '%' in part:
                            result['additive']['proportion'] = float(part.replace('%', '').strip())
                        else:
                            result['additive']['material'] = part.strip()
            
            return result
            
        except Exception as e:
            st.error(f"Failed to parse response: {str(e)}")
            st.write("Raw response:", response_text)
            return None

def predict_composition(properties: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict material composition using Groq API
    """
    # Create a prompt for the LLM
    properties_text = "\n".join([f"{k}: {v}" for k, v in properties.items()])
    prompt = f"""You are a materials science expert. Based on the following material properties:
{properties_text}

Predict the optimal material composition that would exhibit these properties. 
Format your response as a JSON object with the following structure:
{{
    "matrix": {{"material": "material_name", "proportion": percentage_value}},
    "reinforcement": {{"material": "material_name", "proportion": percentage_value}},
    "additive": {{"material": "material_name", "proportion": percentage_value}}
}}

Ensure all proportion values sum to 100. Only include additive if necessary. Keep material names simple and specific.
"""

    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a materials science expert. Provide precise material composition predictions in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3
        )

        # Get the response text
        response_text = chat_completion.choices[0].message.content
        
        # Parse the response
        prediction = parse_llm_response(response_text)
        
        if prediction:
            # Validate the prediction
            total_proportion = (prediction.get('matrix', {}).get('proportion', 0) +
                             prediction.get('reinforcement', {}).get('proportion', 0) +
                             prediction.get('additive', {}).get('proportion', 0))
            
            if not (99 <= total_proportion <= 101):  # Allow for small floating point differences
                st.warning(f"Warning: Total proportion ({total_proportion}%) is not 100%")
            
            return prediction
        else:
            st.error("Could not parse the prediction into the expected format")
            return None
            
    except Exception as e:
        st.error(f"Error during API call: {str(e)}")
        return None

def main():
    st.title("Material Composition Predictor")
    st.write("Enter material properties to predict optimal composition")

    # Create input fields
    with st.form("property_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tensile_strength = st.number_input(
                "Tensile Strength (MPa)",
                min_value=0.0,
                value=200.0
            )
            thermal_conductivity = st.number_input(
                "Thermal Conductivity (W/mK)",
                min_value=0.0,
                value=50.0
            )
            density = st.number_input(
                "Density (g/cm³)",
                min_value=0.0,
                value=2.7
            )

        with col2:
            strength_to_weight = st.number_input(
                "Strength-to-Weight Ratio",
                min_value=0.0,
                value=100.0
            )
            flexibility = st.slider(
                "Flexibility (1-10)",
                min_value=1,
                max_value=10,
                value=5
            )
            hardness = st.number_input(
                "Hardness (HV)",
                min_value=0.0,
                value=100.0
            )

        submit_button = st.form_submit_button("Predict Composition")

    if submit_button:
        # Prepare properties dictionary
        properties = {
            "Tensile Strength (MPa)": tensile_strength,
            "Thermal Conductivity (W/mK)": thermal_conductivity,
            "Density (g/cm³)": density,
            "Strength-to-Weight Ratio": strength_to_weight,
            "Flexibility": flexibility,
            "Hardness (HV)": hardness
        }

        # Show loading spinner
        with st.spinner("Predicting composition..."):
            prediction = predict_composition(properties)

        if prediction:
            # Display results
            st.subheader("Predicted Composition")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Matrix")
                st.write(f"Material: {prediction['matrix']['material']}")
                st.write(f"Proportion: {prediction['matrix']['proportion']}%")
                
            with col2:
                st.markdown("### Reinforcement")
                st.write(f"Material: {prediction['reinforcement']['material']}")
                st.write(f"Proportion: {prediction['reinforcement']['proportion']}%")
                
            with col3:
                if prediction.get('additive') and prediction['additive']['material']:
                    st.markdown("### Additive")
                    st.write(f"Material: {prediction['additive']['material']}")
                    st.write(f"Proportion: {prediction['additive']['proportion']}%")

            # Add visualization
            proportions = {
                prediction['matrix']['material']: prediction['matrix']['proportion'],
                prediction['reinforcement']['material']: prediction['reinforcement']['proportion']
            }
            if prediction.get('additive') and prediction['additive']['material']:
                proportions[prediction['additive']['material']] = prediction['additive']['proportion']

            st.bar_chart(proportions)

if __name__ == "__main__":
    main()