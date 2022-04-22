"""
Utility functions to deal with the Rust extension
"""

RUST_AVAILABLE = None
try:
    import fastsimrust as fsr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def warn_rust_unavailable():
    if not RUST_AVAILABLE:
        print("Warning! FASTSimRust was requested but it is unavailable.")