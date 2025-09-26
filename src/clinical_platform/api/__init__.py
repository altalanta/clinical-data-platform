"""Clinical Data Platform API module."""

from .endpoints import app
from .models import *
from .middleware import *

__all__ = ["app"]