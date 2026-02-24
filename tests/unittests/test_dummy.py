import pytest
from google.genai import types

def test_print_fields():
    print("FIELDS DECLARED:")
    for field in dir(types.GenerateContentResponseUsageMetadata):
        if not field.startswith('_'):
            print(field)
    print("CONFIG FIELDS:")
    for field in dir(types.GenerateContentConfig):
        if not field.startswith('_'):
            print(field)
