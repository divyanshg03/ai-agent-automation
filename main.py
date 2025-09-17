
import subprocess
subprocess.run(["pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", "pip install --upgrade transformers accelerate bitsandbytes"], shell=True)
import os
import torch
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
except ImportError:
    print("❌ Error: Could not import Agent classes. Make sure 'Utils/Agents.py' exists.")
    sys.exit(1)

# Loading API key from a .env file.
load_dotenv(dotenv_path='.env')

def process_report_text(medical_report: str):
    """
    Takes medical report text, runs it through the AI agents,
    and PRINTS the final diagnosis to the console.
    """
    print("\n🩺 Processing the report with AI specialists... (this may take a moment)")

    agents = {
        "Cardiologist": Cardiologist(medical_report),
        "Psychologist": Psychologist(medical_report),
        "Pulmonologist": Pulmonologist(medical_report)
    }

    def get_response(agent_name, agent):
        response = agent.run()
        return agent_name, response

    responses = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
        for future in as_completed(futures):
            agent_name, response = future.result()
            responses[agent_name] = response

    team_agent = MultidisciplinaryTeam(
        cardiologist_report=responses.get("Cardiologist", "No report generated."),
        psychologist_medical_report=medical_report,  # <-- raw report here
        pulmonologist_report=responses.get("Pulmonologist", "No report generated.")
    )

    print("🤝 Synthesizing reports into a final diagnosis...")
    final_diagnosis = team_agent.run()

    print("\n" + "="*60)
    print("### ✅ Final Diagnosis:")
    print("="*60)
    print(final_diagnosis)
    print("="*60)



def main():
    """
    Main function to run the interactive chatbot loop.
    """
    print("🤖 Medical Diagnosis Chatbot Initialized.")
    print("I can analyze a medical report and provide a synthesized diagnosis from multiple specialists.")
    print("Type 'exit' or 'quit' at any time to end the session.")

    while True:
        print("\n" + "-"*60)
        print("👉 Please paste or type the medical report below.")
        print("   (Type 'END' on a new, empty line when you are finished)")
        print("-"*60)

        report_lines = []
        while True:
            try:
                line = input()
                if line.strip().lower() in ['exit', 'quit']:
                    print("👋 Session ended. Goodbye!")
                    return
                if line.strip().upper() == 'END':
                    break
                report_lines.append(line)
            except EOFError: # Handles Ctrl+D to end input
                break

        medical_report_text = "\n".join(report_lines)

        if not medical_report_text.strip():
            print("\n❌ No report text was entered. Please try again.")
            continue
        
        try:
            process_report_text(medical_report_text)
        except Exception as e:
            print(f"\n❌ An unexpected error occurred during processing: {e}")


if __name__ == "__main__":
    main()