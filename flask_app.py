# flask_app.py
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Import your agent classes
try:
    from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
except ImportError:
    print("❌ Error: Could not import Agent classes. Make sure 'Utils/Agents.py' exists.")
    sys.exit(1)

# Load environment variables
load_dotenv(dotenv_path='.env')

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_medical_report():
    """Analyze a medical report and return results from all specialists."""
    try:
        data = request.get_json()
        medical_report = data.get('medical_report', '') if data else ''

        if not medical_report.strip():
            return jsonify({'error': 'Medical report cannot be empty.'}), 400

        # --- Run Cardiologist & Pulmonologist in parallel (Groq API) ---
        agents = {
            "Cardiologist": Cardiologist(medical_report),
            "Pulmonologist": Pulmonologist(medical_report)
        }

        def get_response(agent_name, agent):
            try:
                response = agent.run()
                return agent_name, response, None
            except Exception as e:
                return agent_name, None, str(e)

        responses, errors = {}, {}

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {executor.submit(get_response, n, a): n for n, a in agents.items()}
            for future in as_completed(futures):
                agent_name, response, error = future.result()
                if error:
                    errors[agent_name] = error
                else:
                    responses[agent_name] = response

        # --- Run Psychologist locally (Gemma-2B + LoRA) ---
        try:
            psychologist_agent = Psychologist(medical_report)
            psychologist_report = psychologist_agent.run()
        except Exception as e:
            errors['Psychologist'] = str(e)
            psychologist_report = "Could not be generated."

        # --- Multidisciplinary Team combining all ---
        try:
            team_agent = MultidisciplinaryTeam(
                cardiologist_report=responses.get("Cardiologist", "Cardiologist analysis failed."),
                psychologist_medical_report=medical_report,
                pulmonologist_report=responses.get("Pulmonologist", "Pulmonologist analysis failed.")
            )
            final_diagnosis = team_agent.run()
        except Exception as e:
            errors['MultidisciplinaryTeam'] = str(e)
            final_diagnosis = "Could not be generated."

        return jsonify({
            'success': True,
            'results': {
                'cardiologist': responses.get("Cardiologist", "Analysis failed."),
                'pulmonologist': responses.get("Pulmonologist", "Analysis failed."),
                'psychologist': psychologist_report,
                'final_diagnosis': final_diagnosis
            },
            'errors': errors if errors else None
        })

    except Exception as e:
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500


@app.route('/api/sample-report', methods=['GET'])
def get_sample_report():
    """Provide a sample medical report to the frontend."""
    try:
        sample_report_path = "Medical Reports/Medical Report - Michael Johnson - Panic Attack Disorder.txt"
        if os.path.exists(sample_report_path):
            with open(sample_report_path, "r", encoding="utf-8") as file:
                sample_report = file.read()
        else:
            # Fallback hardcoded report
            sample_report = """
            Medical Case Report
            Patient ID: 345678
            Name: Michael Johnson
            Age: 29
            Gender: Male
            Date of Report: 2024-09-04

            Chief Complaint:
            Sudden episodes of intense chest pain, palpitations, shortness of breath, dizziness,
            sweating lasting 10-20 minutes, occurring weekly. Feeling of impending doom.

            Medical History:
            Anxiety (CBT + benzodiazepines), GERD (PPIs).
            Lifestyle: high-stress job, irregular exercise, caffeine/alcohol occasional use.
            Medications: Lorazepam, Omeprazole.
            Investigations: ECG, labs, Holter, Echo all normal.
            """
        return jsonify({'sample_report': sample_report})
    except Exception as e:
        return jsonify({'error': f'Error loading sample report: {str(e)}'}), 500


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
