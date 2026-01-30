def decide_identity(person_id: str, similarity_score: float, threshold: float = 0.6):
    if similarity_score >= threshold:
        return person_id
    return "unknown"
