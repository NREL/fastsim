"""
Utilities for Testing
"""
import os
import fastsim.demos.utils as utils

def start_demo_environment():
    original_value = os.getenv(utils.DEMO_TEST_ENV_VAR)
    os.environ[utils.DEMO_TEST_ENV_VAR] = 'False'
    return original_value

def end_demo_test_environment(original_value):
    if original_value is not None:
        os.environ[utils.DEMO_TEST_ENV_VAR] = original_value
    else:
        del os.environ[utils.DEMO_TEST_ENV_VAR]