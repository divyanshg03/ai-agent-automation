import os
import torch
from groq import Groq
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Base Agent Class
# ------------------------
class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None,
                 model_id="openai/gpt-oss-120b",
                 temperature: float = 0.3):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = 7000

        # Load Groq API key
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key and role != "Psychologist":  # Psychologist can be local
            raise ValueError("❌ GROQ_API_KEY not found in .env")

        # Initialize Groq client if not local model
        if role != "Psychologist":
            self.client = Groq(api_key=self.api_key)

        # Build prompt template
        self.prompt_template = self.create_prompt_template()

    def create_prompt_template(self):
        """Create prompt templates for different roles."""
        if self.role == "MultidisciplinaryTeam":
            template = (
                "Act like a multidisciplinary team (Cardiology, Psychology, Pulmonology).\n"
                "Task: Analyze the following three specialty reports and produce a DETAILED report.\n"
                "Each bullet: Issue - Reason (concise, evidence-based).\n\n"
                "Cardiologist Report: {cardiologist_report}\n"
                "Psychologist Report: {psychologist_report}\n"
                "Pulmonologist Report: {pulmonologist_report}\n"
            )
            return PromptTemplate.from_template(template)

        templates = {
            "Cardiologist": (
                "Act like a cardiologist. Review the patient's cardiac workup (ECG, labs, Holter, echo).\n"
                "Return ONLY bullet points: possible cardiac causes AND recommended next steps.\n\n"
                "Medical Report: {medical_report}"
            ),
            "Psychologist": (
                "Act like a psychologist. Review the patient's report for mental health concerns.\n"
                "Return ONLY bullet points: possible psychological issues AND recommended next steps.\n\n"
                "Patient Report: {medical_report}"
            ),
            "Pulmonologist": (
                "Act like a pulmonologist. Review the patient's report for respiratory issues.\n"
                "Return ONLY bullet points: possible pulmonary issues AND recommended next steps.\n\n"
                "Patient Report: {medical_report}"
            ),
        }

        if self.role not in templates:
            raise ValueError(f"Unknown role: {self.role}")

        return PromptTemplate.from_template(templates[self.role])

    def _query_groq(self, prompt: str):
        """Send prompt to Groq LLaMA-3.1 model."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": f"You are a {self.role} medical specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens
        )
        return response.choices[0].message.content

    def run(self):
        """Run the agent with its prompt template."""
        print(f"{self.role} is running...")

        if self.role == "MultidisciplinaryTeam":
            prompt = self.prompt_template.format(
                cardiologist_report=self.extra_info.get("cardiologist_report", ""),
                psychologist_report=self.extra_info.get("psychologist_report", ""),
                pulmonologist_report=self.extra_info.get("pulmonologist_report", ""),
            )
            # Use Groq API for final summary if available
            if hasattr(self, "client"):
                return self._query_groq(prompt)
            return prompt  # fallback
        else:
            prompt = self.prompt_template.format(medical_report=self.medical_report or "")
            if self.role == "Psychologist":
                return self._query_local(prompt)
            else:
                return self._query_groq(prompt)


# ------------------------
# Local Psychologist Subclass
# ------------------------
class Psychologist(Agent):
    def __init__(self, medical_report, adapter_path="psychology_lora", **kwargs):
        super().__init__(medical_report, role="Psychologist", **kwargs)

        # -------------------------
        # 4-bit quantization config with CPU offload
        # -------------------------
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True  # ✅ offload to CPU if VRAM is tight
        )

        # -------------------------
        # Load base model with automatic device mapping
        # -------------------------
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            quantization_config=bnb_config,
            device_map="auto",  # Automatically maps layers to GPU/CPU
            offload_folder="offload"  # optional: stores CPU-offloaded weights on disk
        )

        # -------------------------
        # Load LoRA adapter
        # -------------------------
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        self.model = model

        # -------------------------
        # Load tokenizer
        # -------------------------
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def _query_local(self, prompt: str, max_length=256, temperature=0.7, top_p=0.9):
        """Generate output using local LoRA model with offload-safe settings."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
# ------------------------
# Other specialized subclasses
# ------------------------
class Cardiologist(Agent):
    def __init__(self, medical_report, **kwargs):
        super().__init__(medical_report, "Cardiologist", **kwargs)

class Pulmonologist(Agent):
    def __init__(self, medical_report, **kwargs):
        super().__init__(medical_report, "Pulmonologist", **kwargs)


# ------------------------
# Multidisciplinary Team (auto local Psychologist)
# ------------------------
class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_medical_report, pulmonologist_report,
                 psych_adapter_path="psychology_lora", **kwargs):
        """
        cardiologist_report, pulmonologist_report: strings or outputs from their agents
        psychologist_medical_report: raw patient report to feed local Psychologist
        """
        self.cardiologist_report = cardiologist_report
        self.pulmonologist_report = pulmonologist_report
        self.psychologist_medical_report = psychologist_medical_report
        self.psych_adapter_path = psych_adapter_path
        super().__init__(role="MultidisciplinaryTeam", extra_info={}, **kwargs)

        # Initialize local Psychologist internally
        self.local_psychologist = Psychologist(
            medical_report=self.psychologist_medical_report,
            adapter_path=self.psych_adapter_path
        )

    def run(self):
        """Run the multidisciplinary team analysis."""
        print("MultidisciplinaryTeam is running...")

        # Generate psychologist report using local LoRA model
        psych_report = self.local_psychologist._query_local(
            self.psychologist_medical_report
        )

        # Format the combined prompt
        prompt = self.prompt_template.format(
            cardiologist_report=self.cardiologist_report,
            psychologist_report=psych_report,
            pulmonologist_report=self.pulmonologist_report,
        )

        # Optionally, use Groq API for final summary if available
        if hasattr(self, "client"):
            return self._query_groq(prompt)
        return prompt  # fallback to formatted prompt only
