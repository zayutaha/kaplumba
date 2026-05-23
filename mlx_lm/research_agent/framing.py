"""Topic framing — classify topic type and build dimension map."""

from .memory import DIMENSION_TEMPLATES
from ._utils import call_cheap, call_model


KNOWN_TYPES = set(DIMENSION_TEMPLATES.keys())


def infer_topic_type(topic: str, model, tokenizer, args,
                     chat_template_kwargs=None) -> str:
    """Use the model to classify the topic type."""
    messages = [
        {"role": "system", "content": f"Classify this topic into exactly one type: {', '.join(sorted(KNOWN_TYPES))}. Output only the type name."},
        {"role": "user", "content": topic},
    ]
    result = call_cheap(messages, max_tokens=16, temp=0.0)
    if not result:
        result = call_model(messages, max_tokens=16, model=model,
                            tokenizer=tokenizer, args=args,
                            chat_template_kwargs=chat_template_kwargs,
                            temp=0.0)
    result = result.strip().lower().rstrip(".")
    if result in KNOWN_TYPES:
        return result
    return "general_topic"


def build_dimension_map(topic_type: str) -> dict:
    """Return the dimension template for a topic type, with 0.0 initial coverage."""
    dims = DIMENSION_TEMPLATES.get(topic_type, DIMENSION_TEMPLATES["general_topic"])
    return {d: 0.0 for d in dims}
