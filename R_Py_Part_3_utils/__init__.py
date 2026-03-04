# Helper package to expose feature_engineering module from R_Py-Part_3
# (Hyphenated directory names can't be imported directly in Python)
import os
import sys
import importlib.util

_fe_path = os.path.join(os.path.dirname(__file__), "..", "R_Py-Part_3", "feature_engineering.py")
_spec = importlib.util.spec_from_file_location("feature_engineering", _fe_path)
feature_engineering = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feature_engineering)
