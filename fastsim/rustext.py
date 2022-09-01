"""
Utility functions to deal with the Rust extension
"""

# Logging
import logging
logger = logging.getLogger(__name__)

RUST_AVAILABLE = None
try:
    import fastsimrust as fsr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def warn_rust_unavailable(context=None):
    if not RUST_AVAILABLE:
        logger.warn("fastsimrust was requested but it is unavailable")
        if context is not None:
            print(f"- context: {context}")
