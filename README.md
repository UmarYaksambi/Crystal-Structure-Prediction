# Crystal Structure Prediction Tool - README

## Overview
This project provides a **Streamlit-based web application** for predicting crystal structures using a trained machine learning model. The application also offers a scientific explanation of the predicted structure by leveraging the **Groq API** for AI-powered explanations.

## Features
- **Prediction of Crystal Structure:** Identify crystal structures (cubic, orthorhombic, rhombohedral, or tetragonal) based on user-provided input parameters.
- **Stability Index Calculation:** Provides a confidence score for the prediction.
- **AI-Powered Explanation:** Generates a concise, technical explanation for the predicted crystal structure.
- **Interactive User Interface:** User-friendly interface to input parameters and view results.
- **Creation of Public Database:**Easy to read and free to access database creation for students.

## Prerequisites
1. **API Key**: Add your Groq API key to `st.secrets` for the application to work.
2. **Python**: Ensure Python 3.7+ is installed.
3. **Supabase**: Create a supabase account and install the dependencies to get your key and url. Add to `st.secrets` along with the Groq API key. 

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/UmarYaksambi/crystal-structure-predictor.git
   cd crystal-structure-predictor
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your **Groq API key and the Supabase URL and API Key** to the `secrets.toml` file in the `.streamlit/` directory:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   # IF DOING IN A TEAM, ADD THE URL OF THE DATABASE ALREADY CREATED 
   SUPABASE_URL = "your-supabase-url" 
   SUPABASE_KEY = "your-supabase-key"
   ```
4. Place the trained model (`model_1.h5`) in the root directory of the project.



## Database Functionality
The application integrates with **Supabase** to create a public database that stores user predictions and related data. This allows users to access historical predictions and their corresponding stability indices.

### Setting Up the Database
1. **Create a Supabase Project**: Sign up at [Supabase](https://supabase.io/) and create a new project.
2. **Configure Database**: Set up the necessary tables to store prediction data. A suggested schema includes:
   - **Predictions Table**:
     - `id`: Unique identifier for each prediction (Primary Key).
     - `input_parameters`: JSON object containing user input parameters.
     - `predicted_structure`: The predicted crystal structure.
     - `stability_index`: Confidence score for the prediction.
     - `timestamp`: Date and time of the prediction.

3. **Connect to Supabase**: Ensure your Supabase URL and API key are added to the `secrets.toml` file as mentioned in the Installation section.

### Accessing the Database
- Users can view their past predictions through the Supabase Website.
- The application allows for the addition of new predictions to the database after each prediction is made.
- It can be programmed for modification and removal of entries from the database as well.



## How to Run
1. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your browser (usually at `http://localhost:8501`).

## Usage
1. Enter the required input parameters, including:
   - Atomic radii, electronegativities, bond lengths, valencies, and elements for A and B sites.
2. Click the **Predict Crystal Structure** button.
3. View the predicted structure, stability index, and an AI-generated explanation.

## File Structure
- **`app.py`**: Main application script.
- **`requirements.txt`**: List of dependencies.
- **`model_1.h5`**: Pre-trained machine learning model (to be added by the user).
- **`database.py`**: The file that connects the application as a client to the database.
- **`.streamlit/secrets.toml`**: File to store API keys securely.

## Dependencies
Install the following Python libraries (handled via `requirements.txt`):
- `numpy`
- `pandas`
- `tensorflow`
- `streamlit`
- `groq`
- `supabase`

## Customization
- Update the `model_1.h5` file with a different trained model if needed.
- Modify `prepare_feature_vector` to match the feature structure of your dataset.

## API Reference
### Groq API
The application uses the **Groq API** to generate explanations for the predicted structure. Ensure your API key has access to the required endpoints.

## Contributing
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
[☕ Buy Me a Coffee](https://www.buymeacoffee.com/umaryaksambi)

For questions or support, contact **Umar Yaksambi** at [umaryaksambi@gmail.com](mailto:umaryaksambi@gmail.com).
