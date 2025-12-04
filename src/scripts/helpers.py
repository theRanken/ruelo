def base64_string_looks_like_image(s):
    """
        Helper function to determine if a long string is likely a base64 encoded image.
        This is a heuristic and not foolproof.
        """
    # Check for common base64 characters and length
    return len(s) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s[-100:])


# Map similarity to a confidence level label
def get_confidence_label(similarity, backend):
    if backend in ["arcface", "facenet", "ultraface"]:
        if similarity >= 0.90:
            return "very_high"
        elif similarity >= 0.85:
            return "high"
        elif similarity >= 0.80:
            return "medium"
        else:
            return "low"
    elif backend == "sface":
        if similarity <= 0.30:
            return "very_high"
        elif similarity <= 0.35:
            return "high"
        elif similarity <= 0.40:
            return "medium"
        else:
            return "low"
    else:
        return "unknown"
