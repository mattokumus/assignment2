#!/usr/bin/env python3
"""
JSON Automatic Documentation Generator
This script analyzes JSON files and generates detailed documentation.
"""

import json
from typing import Any, Dict, List, Set
from collections import defaultdict
import sys


class JSONDocumenter:
    def __init__(self):
        self.field_types = defaultdict(set)
        self.field_examples = defaultdict(list)
        self.field_paths = []
        self.field_frequency = defaultdict(int)  # Counts how many times it was seen
        self.total_objects_seen = defaultdict(int)  # How many objects seen at each level
        
    def analyze_value(self, value: Any, path: str = "root"):
        """Analyzes the type and example values of a value"""
        
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            # Detect string formats
            if value.startswith("http://") or value.startswith("https://"):
                return "string (URL)"
            elif "@" in value and "." in value:
                return "string (email?)"
            elif "/" in value and len(value) == 10:
                return "string (date?)"
            else:
                return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return type(value).__name__
    
    def traverse_json(self, obj: Any, path: str = "root", parent_is_array: bool = False, max_samples: int = 100, parent_path: str = ""):
        """Recursively traverses JSON structure and collects information

        Args:
            max_samples: How many samples to analyze in arrays (default: 100)
            parent_path: Parent level path (for optional field detection)
        """
        
        if isinstance(obj, dict):
            # We saw an object at this level
            if parent_path:
                self.total_objects_seen[parent_path] += 1

            for key, value in obj.items():
                current_path = f"{path}.{key}" if path != "root" else key

                # We saw this field, increment counter
                self.field_frequency[current_path] += 1
                
                value_type = self.analyze_value(value)
                self.field_types[current_path].add(value_type)

                # Record example value (first 3 examples)
                if len(self.field_examples[current_path]) < 3:
                    if not isinstance(value, (dict, list)):
                        self.field_examples[current_path].append(value)
                    elif isinstance(value, list) and len(value) > 0:
                        self.field_examples[current_path].append(f"Array with {len(value)} items")
                    elif isinstance(value, dict):
                        self.field_examples[current_path].append(f"Object with {len(value)} keys")

                # Recursive traversal
                if isinstance(value, dict):
                    self.traverse_json(value, current_path, False, max_samples, current_path)
                elif isinstance(value, list) and len(value) > 0:
                    # Analyze multiple elements of array
                    items_to_check = min(len(value), max_samples)
                    for i in range(items_to_check):
                        item = value[i]
                        if isinstance(item, dict):
                            self.traverse_json(item, f"{current_path}[]", True, max_samples, f"{current_path}[]")
                        else:
                            item_type = self.analyze_value(item)
                            self.field_types[f"{current_path}[]"].add(item_type)
                            if len(self.field_examples[f"{current_path}[]"]) < 3:
                                self.field_examples[f"{current_path}[]"].append(item)
        
        elif isinstance(obj, list):
            # If there is an array at the main level
            if len(obj) > 0:
                items_to_check = min(len(obj), max_samples)
                for i in range(items_to_check):
                    item = obj[i]
                    if isinstance(item, dict):
                        # For root level array, parent_path should be "root"
                        self.traverse_json(item, path if path != "root" else "root", True, max_samples, "root")
                    else:
                        item_type = self.analyze_value(item)
                        self.field_types[path].add(item_type)
                        if len(self.field_examples[path]) < 3:
                            self.field_examples[path].append(item)
    
    def generate_markdown(self, title: str = "JSON Documentation") -> str:
        """Generates documentation in Markdown format"""

        md = f"# {title}\n\n"
        md += f"**Creation Date:** {self._get_current_date()}\n\n"
        md += "---\n\n"

        md += "## Contents\n\n"
        md += "1. [Overview](#overview)\n"
        md += "2. [Field Details](#field-details)\n"
        md += "3. [Data Structure](#data-structure)\n\n"

        md += "---\n\n"

        # Overview
        md += "## Overview\n\n"
        md += f"**Total Number of Fields:** {len(self.field_types)}\n\n"
        
        type_counts = defaultdict(int)
        for types in self.field_types.values():
            for t in types:
                type_counts[t] += 1

        md += "**Type Distribution:**\n\n"
        for tip, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            md += f"- {tip}: {count} fields\n"

        md += "\n---\n\n"

        # Field Details
        md += "## Field Details\n\n"

        # Sort paths
        sorted_paths = sorted(self.field_types.keys())
        
        for path in sorted_paths:
            types = self.field_types[path]
            examples = self.field_examples.get(path, [])
            frequency = self.field_frequency.get(path, 0)

            # Header size according to path level
            level = path.count('.') + path.count('[]')
            header_level = min(level + 3, 6)

            md += f"{'#' * header_level} `{path}`\n\n"

            # Type information
            type_str = " | ".join(sorted(types))
            md += f"**Type:** {type_str}\n\n"

            # Optional/Required information
            # Check "root" parent for root level fields
            parent_path = ""
            if '.' in path:
                parent_path = ".".join(path.split('.')[:-1])
            elif '[]' not in path:  # Root level fields
                parent_path = "root"

            if parent_path in self.total_objects_seen:
                total = self.total_objects_seen[parent_path]
                if frequency == total:
                    md += f"**Status:** ✅ Required (present in all {total} examples)\n\n"
                else:
                    percentage = (frequency / total * 100) if total > 0 else 0
                    md += f"**Status:** ⚠️ Optional (present in {frequency}/{total} examples, {percentage:.1f}%)\n\n"

            # Example values
            if examples:
                md += "**Example Values:**\n\n"
                for i, example in enumerate(examples[:3], 1):
                    if isinstance(example, str) and len(example) > 100:
                        example = example[:100] + "..."
                    md += f"{i}. `{example}`\n"
                md += "\n"
            
            md += "---\n\n"
        
        return md
    
    def generate_json_schema(self, title: str = "JSON Schema", description: str = "") -> str:
        """JSON Schema formatında dokümantasyon üretir"""
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": title,
            "description": description or "Auto-generated JSON Schema",
            "type": "object",
            "properties": {}
        }
        
        # Root level array ise bunu belirt
        root_keys = [k for k in self.field_types.keys() if '.' not in k and '[]' not in k]
        
        # Eğer root'ta sadece array elemanları varsa, schema'yı array olarak düzenle
        if all('[]' in k for k in self.field_types.keys() if '.' not in k):
            schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": title,
                "description": description or "Auto-generated JSON Schema",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {}
                }
            }
            target_properties = schema["items"]["properties"]
        else:
            target_properties = schema["properties"]
        
        # Her path'i işle
        processed_paths = set()
        
        for path in sorted(self.field_types.keys()):
            if path in processed_paths:
                continue
            
            # Root level alanları işle
            if '.' not in path and '[]' not in path:
                field_schema = self._build_field_schema(path)
                target_properties[path] = field_schema
                processed_paths.add(path)
        
        return json.dumps(schema, indent=2, ensure_ascii=False)
    
    def _build_field_schema(self, path: str) -> dict:
        """Creates schema for a single field"""
        types = self.field_types.get(path, set())
        examples = self.field_examples.get(path, [])
        
        field_schema = {}

        # Type determination
        type_mapping = {
            "string": "string",
            "string (URL)": "string",
            "string (email?)": "string",
            "string (date?)": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "null": "null",
            "array": "array",
            "object": "object"
        }
        
        json_types = [type_mapping.get(t, "string") for t in types if t in type_mapping]
        
        if len(json_types) == 1:
            field_schema["type"] = json_types[0]
        elif len(json_types) > 1:
            # Use anyOf if there are multiple types
            field_schema["anyOf"] = [{"type": t} for t in json_types]

        # Add format information
        for t in types:
            if "URL" in t:
                field_schema["format"] = "uri"
            elif "email" in t:
                field_schema["format"] = "email"
            elif "date" in t:
                field_schema["format"] = "date"

        # If array, add items information
        if "array" in json_types:
            array_item_path = f"{path}[]"

            # Check array sub-fields (does it contain objects?)
            array_children = [p for p in self.field_types.keys() if p.startswith(f"{path}[].")]

            if array_children:
                # There is an object in the array
                field_schema["items"] = {
                    "type": "object",
                    "properties": {}
                }
                # Add sub-fields of the object in the array
                for sub_path in array_children:
                    # Skip array items paths (ending with [])
                    if sub_path.endswith('[]'):
                        continue
                    if sub_path.count('.') == path.count('.') + 1:
                        field_name = sub_path.split('.')[-1]
                        field_schema["items"]["properties"][field_name] = self._build_field_schema(sub_path)
            elif array_item_path in self.field_types:
                # There is a primitive type in the array
                item_types = self.field_types[array_item_path]
                item_json_types = [type_mapping.get(t, "string") for t in item_types if t in type_mapping]
                if len(item_json_types) == 1:
                    field_schema["items"] = {"type": item_json_types[0]}
                elif len(item_json_types) > 1:
                    field_schema["items"] = {"anyOf": [{"type": t} for t in item_json_types]}

        # If object, add properties information
        if "object" in json_types:
            # Find sub-fields of this object
            child_paths = [p for p in self.field_types.keys() if p.startswith(f"{path}.") and p != path]

            if child_paths:
                # Immediate children = paths that are only one level deep
                # To find this: examine child_paths
                immediate_children = []
                for p in child_paths:
                    # Get the part after path
                    suffix = p[len(path) + 1:]  # +1 for the dot

                    # Check if '[]' is in suffix
                    if '[]' in suffix:
                        # Does suffix start with '[]'?
                        if suffix == '[]' or suffix.startswith('[].'):
                            continue  # This is not immediate, it's an array element
                        # Suffix format: can be 'key[]' or 'key[].something'
                        # 'key[]' is immediate, 'key[].something' is not
                        bracket_pos = suffix.index('[]')
                        before_bracket = suffix[:bracket_pos]
                        after_bracket = suffix[bracket_pos + 2:]
                        
                        if '.' not in before_bracket and after_bracket == '':
                            immediate_children.append(p)
                    else:
                        # No '[]' - normal key
                        # Check for deeper paths: only count paths continuing with '.'
                        # Those continuing with '[]' are at the same level
                        has_deeper = any(cp.startswith(p + '.') and not cp.startswith(p + '[]')
                                       for cp in child_paths if cp != p)
                        if not has_deeper:
                            immediate_children.append(p)

                if immediate_children:
                    # Filter array items paths (remove those ending with [])
                    # Example: 'details[]' is not a property, it's items of 'details' array
                    immediate_children = [p for p in immediate_children if not p.endswith('[]')]

                if immediate_children:
                    # Get all keys
                    all_keys = [p[len(path) + 1:] for p in immediate_children]
                    # Clean keys containing '[]' (get only the key part)
                    all_keys = [k.split('[]')[0] if '[]' in k else k for k in all_keys]
                    unique_keys = set(all_keys)

                    # If keys have file extensions (.docx, .pdf, etc.) or
                    # many different keys, it might be dynamic
                    file_extensions = ['.docx', '.pdf', '.xlsx', '.txt', '.json', '.xml', '.html']
                    looks_dynamic = any(any(k.endswith(ext) for ext in file_extensions) for k in unique_keys)

                    if looks_dynamic and len(immediate_children) > 0:
                        # Dynamic key pattern - take a representative path
                        representative_path = immediate_children[0]
                        field_schema["additionalProperties"] = self._build_field_schema(representative_path)
                    else:
                        # Fixed keys - normal properties
                        field_schema["properties"] = {}
                        for sub_path in immediate_children:
                            # Extract key name
                            field_name = sub_path[len(path) + 1:]
                            if '[]' in field_name:
                                field_name = field_name.split('[]')[0]
                            if field_name not in field_schema["properties"]:
                                field_schema["properties"][field_name] = self._build_field_schema(sub_path)
            else:
                # No sub-fields, empty properties
                field_schema["properties"] = {}

        # Add example values
        if examples:
            # Filter examples that are strings and don't start with "Array" or "Object"
            clean_examples = []
            for ex in examples[:3]:
                if isinstance(ex, str):
                    if not ex.startswith("Array") and not ex.startswith("Object"):
                        clean_examples.append(ex)
                else:
                    clean_examples.append(ex)
            
            if clean_examples:
                field_schema["examples"] = clean_examples
        
        return field_schema
    
    def generate_html(self, title: str = "JSON Documentation") -> str:
        """Generates documentation in HTML format"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .field {{
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        .field-path {{
            font-family: 'Courier New', monospace;
            color: #e74c3c;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .field-type {{
            color: #27ae60;
            font-weight: bold;
        }}
        .example {{
            background: white;
            padding: 8px;
            margin: 5px 0;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .stats {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p><strong>Creation Date:</strong> {self._get_current_date()}</p>

        <div class="stats">
            <h2>Overview</h2>
            <p><strong>Total Number of Fields:</strong> {len(self.field_types)}</p>
        </div>

        <h2>Field Details</h2>
"""
        
        sorted_paths = sorted(self.field_types.keys())
        
        for path in sorted_paths:
            types = self.field_types[path]
            examples = self.field_examples.get(path, [])
            
            type_str = " | ".join(sorted(types))
            
            html += f"""
        <div class="field">
            <div class="field-path">{path}</div>
            <p><span class="field-type">Type:</span> {type_str}</p>
"""

            if examples:
                html += "            <p><strong>Example Values:</strong></p>\n"
                for example in examples[:3]:
                    if isinstance(example, str) and len(example) > 100:
                        example = example[:100] + "..."
                    html += f'            <div class="example">{self._escape_html(str(example))}</div>\n'
            
            html += "        </div>\n"
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def _escape_html(self, text: str) -> str:
        """Escapes HTML characters"""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    def _get_current_date(self) -> str:
        """Returns current date"""
        from datetime import datetime
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def main():
    # Default to cases-2000.json if no argument provided
    if len(sys.argv) < 2:
        json_file = "cases-2000.json"
        output_format = "all"  # default format
        print(f"No arguments provided. Using default file: {json_file}")
        print("Usage: python jsondocumenting.py <json_file> [--format md|html|schema|both|all]")
        print("\nExamples:")
        print("  python jsondocumenting.py data.json")
        print("  python jsondocumenting.py data.json --format html")
        print("  python jsondocumenting.py data.json --format schema")
        print("  python jsondocumenting.py data.json --format both     # md + html")
        print("  python jsondocumenting.py data.json --format all      # md + html + schema")
        print()
    else:
        json_file = sys.argv[1]
        output_format = "md"  # default
    
    if len(sys.argv) > 2 and sys.argv[2] == "--format":
        if len(sys.argv) > 3:
            output_format = sys.argv[3].lower()
    
    # Read JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read JSON file: {e}")
        sys.exit(1)

    # Analyze
    print("Analyzing JSON file...")
    documenter = JSONDocumenter()
    documenter.traverse_json(data)

    # Generate documentation
    base_name = json_file.rsplit('.', 1)[0]

    if output_format in ["md", "both", "all"]:
        md_content = documenter.generate_markdown()
        md_file = f"{base_name}_documentation.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Markdown documentation created: {md_file}")

    if output_format in ["html", "both", "all"]:
        html_content = documenter.generate_html()
        html_file = f"{base_name}_documentation.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ HTML documentation created: {html_file}")

    if output_format in ["schema", "all"]:
        schema_content = documenter.generate_json_schema()
        schema_file = f"{base_name}_schema.json"
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write(schema_content)
        print(f"✓ JSON Schema created: {schema_file}")

    print(f"\nTotal {len(documenter.field_types)} fields analyzed.")


if __name__ == "__main__":
    main()