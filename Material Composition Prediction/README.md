# Material Composition Predictor

A Streamlit application that predicts optimal material compositions based on input properties using the Groq API and the Mixtral-8x7b model. This tool helps materials scientists and engineers quickly estimate material compositions that would achieve desired physical properties.

## Features

- Predicts matrix, reinforcement, and additive materials with their proportions
- Handles multiple material properties as input:
  - Tensile Strength (MPa)
  - Thermal Conductivity (W/mK)
  - Density (g/cm³)
  - Strength-to-Weight Ratio
  - Flexibility (1-10 scale)
  - Hardness (HV)
- Provides visual representation of composition through bar charts
- Robust error handling and response parsing
- User-friendly interface built with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/UmarYaksambi/crystal-structure-predictor.git
cd crystal-structure-predictor/Material\ Composition\ Prediction
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file in the project directory:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Replace `your-groq-api-key-here` with your actual Groq API key. If you don't have one, you can obtain it from [Groq's website](https://www.groq.com).

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Enter the desired material properties in the input fields:
   - Input the values for each property
   - Use the slider for flexibility rating
   - Click "Predict Composition" to generate results

4. View the results:
   - Matrix material and proportion
   - Reinforcement material and proportion
   - Additive material and proportion (if applicable)
   - Visual representation of the composition distribution

## Dependencies

- streamlit
- groq

## Error Handling

The application includes robust error handling for:
- API response parsing
- Invalid input validation
- Connection issues
- Proportion validation (ensures components sum to 100%)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the Groq API and Mixtral-8x7b model
- Streamlit for the web interface
- Materials science principles for prediction logic

## Support

If you encounter any issues or have questions, please:
1. Check the existing issues in the GitHub repository
2. Create a new issue if your problem hasn't been reported
3. Provide detailed information about your environment and the error

## Future Improvements

- [ ] Add more material properties as input parameters
- [ ] Implement result caching for similar inputs
- [ ] Add confidence scores for predictions
- [ ] Include more detailed material property explanations
- [ ] Add export functionality for results
- [ ] Implement batch processing capabilities

---
Created by [Umar Yaksambi](https://github.com/UmarYaksambi)
