"""Exceptions for the aigym package."""


class NoPathsFoundError(Exception):
    """Exception raised when no paths are found in a web page."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"No links found in {url}.")
