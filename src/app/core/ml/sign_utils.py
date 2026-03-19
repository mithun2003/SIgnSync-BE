"""Utility functions for sign language processing and formatting."""

# Display name mapping: internal model name → user-friendly display name
SIGN_DISPLAY_NAMES = {
    # Emergency/Custom signs
    "ok_sign": "Ok",
    "thumbs_down": "Not Ok",
    "help": "Help",
    "danger": "Danger",
    "emergency": "Emergency",
    # Standard ASL alphabet + space/del (capitalize single letters)
}


def format_sign_name(sign: str) -> str:
    """Convert internal sign name to user-friendly display format.

    Args:
        sign: Internal sign name (e.g., "ok_sign", "thumbs_down", "A")

    Returns:
        User-friendly display name (e.g., "Ok", "Not Ok", "A")

    Examples:
        >>> format_sign_name("ok_sign")
        'Ok'
        >>> format_sign_name("thumbs_down")
        'Not Ok'
        >>> format_sign_name("A")
        'A'
        >>> format_sign_name("space")
        'Space'
    """
    if sign in SIGN_DISPLAY_NAMES:
        return SIGN_DISPLAY_NAMES[sign]
    # For standard letters and del/space, capitalize
    if sign.lower() in ["del", "space"]:
        return sign.capitalize()
    # Single uppercase letter (A-Z)
    if len(sign) == 1 and sign.isalpha():
        return sign.upper()
    # Fallback: return as-is
    return sign
