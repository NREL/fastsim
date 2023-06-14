"""
Utility functions for demo code
"""
DEMO_TEST_ENV_VAR = 'FASTSIM_DEMO_IS_INTERACTIVE'

def maybe_str_to_bool(x, default=True):
    """
    Turn values of None or string to bool
    - x: str | None, some parameter, a string or None
    - default: Bool, the default if x is None or blank
    RETURN: True or False
    """
    if x is None:
        return default
    if x is True or x is False:
        return x
    try:
        lower_cased = x.lower().strip()
        if lower_cased == 'false':
            return False
        if lower_cased == 'true':
            return True
        return default
    except:
        return default
