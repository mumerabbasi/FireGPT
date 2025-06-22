"""A client library for accessing Forest Fire Danger Assessment API"""

from .client import AuthenticatedClient, Client

__all__ = (
    "AuthenticatedClient",
    "Client",
)
