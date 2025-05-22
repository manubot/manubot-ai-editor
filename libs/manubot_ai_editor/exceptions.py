"""
Exception classes that are shared across modules in the project.
"""


class APIKeyInvalidError(Exception):
    """
    Raised when a provider request is attempted with an invalid API key.
    """

    pass
