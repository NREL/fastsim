"""
Utility functions to deal with the Rust extension
"""

RUST_AVAILABLE = None
try:
    import fastsimrust as fsr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
