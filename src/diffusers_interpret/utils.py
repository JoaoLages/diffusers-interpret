def clean_token_from_prefixes_and_suffixes(token: str) -> str:
    """
    Removes all the known token prefixes and suffixes

    Args:
        token (`str`): string with token

    Returns:
        `str`: clean token
    """

    # removes T5 prefix
    token = token.lstrip('▁')

    # removes BERT/GPT-2 prefix
    token = token.lstrip('Ġ')

    # removes CLIP suffix
    token = token.rstrip('</w>')

    return token