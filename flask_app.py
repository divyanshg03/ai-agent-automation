# flask_app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import json

# Import your existing agents
try:
    from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
except ImportError:
    print("Error: Could not import Agent classes. Make sure 'Utils/Agents.py' exists.")
    sys.exit(1)

# Load environment variables
load_dotenv(dotenv_path='.env')
  
app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_medical_report():
    """API endpoint to analyze medical reports"""
    try:
        # Get the medical report from the request
        data = request.get_json()
        medical_report = data.get('medical_report', '')
        
        if not medical_report.strip():
            return jsonify({'error': 'Medical report is required'}), 400

        # Initialize the specialist agents
        agents = {
            "Cardiologist": Cardiologist(medical_report),
            "Psychologist": Psychologist(medical_report),
            "Pulmonologist": Pulmonologist(medical_report)
        }

        # Function to run each agent and get their response
        def get_response(agent_name, agent):
            try:
                response = agent.run()
                return agent_name, response, None
            except Exception as e:
                return agent_name, None, str(e)

        # Run agents concurrently and collect their reports
        responses = {}
        errors = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
            
            for future in as_completed(futures):
                agent_name, response, error = future.result()
                if error:
                    errors[agent_name] = error
                else:
                    responses[agent_name] = response

        # Check if we have any successful responses
        if not responses:
            return jsonify({'error': 'All agents failed to process the report', 'details': errors}), 500

        # Initialize the multidisciplinary team
        team_agent = MultidisciplinaryTeam(
            cardiologist_report=responses.get("Cardiologist", "Analysis failed"),
            psychologist_report=responses.get("Psychologist", "Analysis failed"),
            pulmonologist_report=responses.get("Pulmonologist", "Analysis failed")
        )

        # Generate final diagnosis
        try:
            final_diagnosis = team_agent.run()
        except Exception as e:
            final_diagnosis = f"Error generating final diagnosis: {str(e)}"

        # Return the results
        return jsonify({
            'success': True,
            'results': {
                'cardiologist': responses.get("Cardiologist", "Analysis failed"),
                'psychologist': responses.get("Psychologist", "Analysis failed"),
                'pulmonologist': responses.get("Pulmonologist", "Analysis failed"),
                'final_diagnosis': final_diagnosis
            },
            'errors': errors if errors else None
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/sample-report', methods=['GET'])
def get_sample_report():
    """Get the sample medical report"""
    try:
        sample_report_path = "Medical Reports/Medical Report - Michael Johnson - Panic Attack Disorder.txt"
        if os.path.exists(sample_report_path):
            with open(sample_report_path, "r", encoding="utf-8") as file:
                sample_report = file.read()
            return jsonify({'sample_report': sample_report})
        else:
            # Fallback sample report
            fallback_report = """Medical Case Report
Patient ID: 345678
Name: Michael Johnson
Age: 29
Gender: Male
Date of Report: 2024-09-04

Chief Complaint:
The patient reports experiencing sudden episodes of intense chest pain, heart palpitations, shortness of breath, dizziness, and sweating over the past three months. These episodes typically last for about 10-20 minutes and occur without warning, usually once or twice a week. The patient describes a feeling of impending doom during these episodes and fears having a heart attack.

Medical History:
Family History: No known history of heart disease. Mother has generalized anxiety disorder; father has no significant medical history.
Personal Medical History:
Anxiety: Diagnosed at age 25; managed with cognitive behavioral therapy (CBT) and occasional use of benzodiazepines.
Gastroesophageal Reflux Disease (GERD): Diagnosed at age 27; managed with proton pump inhibitors (PPIs) and dietary changes.
Lifestyle Factors: The patient works in a high-stress job as an investment banker, reports occasional use of caffeine and alcohol, and exercises irregularly.
Medications: Lorazepam (0.5 mg as needed for anxiety), Omeprazole (20 mg daily).

Recent Lab and Diagnostic Results:
Electrocardiogram (ECG): Normal sinus rhythm; no signs of ischemia or arrhythmia detected.
Blood Tests: Cardiac enzymes (troponin, CK-MB) within normal limits; thyroid function tests normal.
Holter Monitor (24-hour monitoring): No significant arrhythmias; occasional premature ventricular contractions (PVCs) noted.
Echocardiogram: Normal cardiac structure and function; ejection fraction 60%.

Physical Examination Findings:
Vital Signs: Blood pressure 122/78 mmHg, heart rate 82 bpm, BMI 23.4.
Cardiovascular Exam: Normal heart sounds; no murmurs, gallops, or rubs.
Respiratory Exam: Clear breath sounds; no wheezing or crackles."""
            return jsonify({'sample_report': fallback_report})
    except Exception as e:
        return jsonify({'error': f'Error loading sample report: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)