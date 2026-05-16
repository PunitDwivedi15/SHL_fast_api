"""
fix_catalog.py — Fixes literal newlines INSIDE JSON string values.
The catalog has entries like "Microsoft\n(literal newline)Word" inside strings,
which is invalid JSON. This script removes them.
"""

import re

print("Reading raw catalog file...")

with open("catalog.json", "r", encoding="utf-8") as f:
    raw = f.read()

print(f"File size: {len(raw)} characters")

# This function processes the raw text character by character.
# When we're INSIDE a JSON string (between quotes), we remove or replace
# any literal control characters including newlines.
# When we're OUTSIDE a string, we leave newlines alone (they're fine there).

def clean_json_strings(text):
    result = []
    inside_string = False
    i = 0
    
    while i < len(text):
        ch = text[i]
        
        # Track whether we're inside a JSON string
        # A quote toggles us in/out, UNLESS it's escaped with backslash
        if ch == '"' and (i == 0 or text[i-1] != '\\'):
            inside_string = not inside_string
            result.append(ch)
        
        elif inside_string:
            # Inside a string: remove or replace control characters
            if ch == '\n':
                # Replace literal newline with a space
                result.append(' ')
            elif ch == '\r':
                # Remove carriage return
                pass
            elif ch == '\t':
                # Replace tab with space
                result.append(' ')
            elif ord(ch) < 32:
                # Remove any other control character
                pass
            else:
                result.append(ch)
        
        else:
            # Outside a string: keep everything as-is
            result.append(ch)
        
        i += 1
    
    return ''.join(result)

print("Cleaning control characters inside strings...")
cleaned = clean_json_strings(raw)

print("Parsing cleaned JSON...")

import json

try:
    catalog = json.loads(cleaned)
    print(f"✓ Success! Loaded {len(catalog)} items.")
    
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    
    print("✓ Saved clean catalog.json")
    print("Now run: python build_index.py")

except json.JSONDecodeError as e:
    print(f"✗ Still failing: {e}")
    lines = cleaned.split('\n')
    if e.lineno <= len(lines):
        problem_line = lines[e.lineno - 1]
        start = max(0, e.colno - 60)
        end = min(len(problem_line), e.colno + 60)
        print(f"\nProblematic area (line {e.lineno}, col {e.colno}):")
        print(repr(problem_line[start:end]))