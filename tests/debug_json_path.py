
import json
from pathlib import Path

# Mock _truncate_strings from judger_pipeline.py
def _truncate_strings(value, max_len=2000):
    if isinstance(value, Path):
        print(f"DEBUG: Found Path object: {value}")
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_truncate_strings(item, max_len=max_len) for item in value]
    if isinstance(value, dict):
        return {k: _truncate_strings(v, max_len=max_len) for k, v in value.items()}
    return value

def test_truncation():
    print("--- Testing _truncate_strings ---")
    data = {
        "simple_path": Path("C:/foo"),
        "nested": {"p": Path("D:/bar")},
        "list": [Path("E:/baz")],
        "tuple_warning": (Path("F:/qux"),), # Tuples are NOT handled by _truncate_strings!
    }
    
    truncated = _truncate_strings(data)
    print("Truncated Result:", truncated)
    
    try:
        json.dumps(truncated)
        print("✅ JSON Serialization SUCCESS")
    except TypeError as e:
        print(f"❌ JSON Serialization FAILED: {e}")

    print("\n--- Testing default=str ---")
    try:
        json.dumps(data, default=str)
        print("✅ JSON Serialization with default=str SUCCESS")
    except TypeError as e:
        print(f"❌ JSON Serialization with default=str FAILED: {e}")

if __name__ == "__main__":
    test_truncation()
