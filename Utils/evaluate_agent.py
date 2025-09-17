# utils/evaluate_agent.py
import spacy
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease

# Load models once for efficiency
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Use MedSpaCy if installed, else fallback to general English model
try:
    nlp = spacy.load("en_core_sci_sm")  # med-specific model
except:
    nlp = spacy.load("en_core_web_sm")


def semantic_similarity(input_text: str, agent_output: str) -> float:
    """
    Compute cosine similarity between input report and agent output.
    Returns a score between 0 and 1.
    """
    emb_input = embedding_model.encode(input_text, convert_to_tensor=True)
    emb_output = embedding_model.encode(agent_output, convert_to_tensor=True)
    return util.cos_sim(emb_input, emb_output).item()


def entity_coverage(input_text: str, agent_output: str) -> float:
    """
    Compute proportion of medical entities in input that appear in the agent output.
    Returns a score between 0 and 1.
    """
    doc_input = nlp(input_text)
    doc_output = nlp(agent_output)

    input_entities = {ent.text.lower() for ent in doc_input.ents if ent.label_ in ["DISEASE", "SYMPTOM", "MEDICATION", "CONDITION"]}
    output_entities = {ent.text.lower() for ent in doc_output.ents if ent.label_ in ["DISEASE", "SYMPTOM", "MEDICATION", "CONDITION"]}

    if not input_entities:
        return 1.0  # Nothing to cover, assume full coverage

    matched = input_entities.intersection(output_entities)
    return len(matched) / len(input_entities)


def readability_score(agent_output: str) -> float:
    """
    Compute Flesch Reading Ease for the agent output.
    Higher is easier to read.
    """
    return flesch_reading_ease(agent_output)


def evaluate_agent(input_text: str, agent_output: str) -> dict:
    """
    Evaluate agent output using multiple metrics.
    Returns a dictionary with scores.
    """
    sim = semantic_similarity(input_text, agent_output)
    coverage = entity_coverage(input_text, agent_output)
    readability = readability_score(agent_output)

    # Composite score (weights can be adjusted)
    composite = 0.4 * sim + 0.4 * coverage + 0.2 * (readability / 100)  # readability scaled to 0-1

    return {
        "semantic_similarity": sim,
        "entity_coverage": coverage,
        "readability_score": readability,
        "composite_score": composite
    }


if __name__ == "__main__":
    # Quick test
    input_text = "Patient reports anxiety, poor sleep, and rapid heartbeat."
    agent_output = "Likely anxiety disorder. Recommend CBT and relaxation techniques."

    scores = evaluate_agent(input_text, agent_output)
    print("Evaluation Metrics:", scores)
 