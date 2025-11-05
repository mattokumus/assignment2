#!/usr/bin/env python3
"""
JSON Otomatik Dokümantasyon Oluşturucu
Bu script JSON dosyalarını analiz eder ve detaylı dokümantasyon üretir.
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
        self.field_frequency = defaultdict(int)  # Kaç kez görüldüğünü sayar
        self.total_objects_seen = defaultdict(int)  # Her seviyede kaç obje görüldü
        
    def analyze_value(self, value: Any, path: str = "root"):
        """Bir değerin tipini ve örnek değerlerini analiz eder"""
        
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            # String formatlarını tespit et
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
        """JSON yapısını recursive olarak gezer ve bilgi toplar
        
        Args:
            max_samples: Array'lerde kaç örnek analiz edilecek (varsayılan: 100)
            parent_path: Üst seviye path (opsiyonel alan tespiti için)
        """
        
        if isinstance(obj, dict):
            # Bu seviyede bir object gördük
            if parent_path:
                self.total_objects_seen[parent_path] += 1
            
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path != "root" else key
                
                # Bu alanı gördük, sayacı artır
                self.field_frequency[current_path] += 1
                
                value_type = self.analyze_value(value)
                self.field_types[current_path].add(value_type)
                
                # Örnek değer kaydet (ilk 3 örnek)
                if len(self.field_examples[current_path]) < 3:
                    if not isinstance(value, (dict, list)):
                        self.field_examples[current_path].append(value)
                    elif isinstance(value, list) and len(value) > 0:
                        self.field_examples[current_path].append(f"Array with {len(value)} items")
                    elif isinstance(value, dict):
                        self.field_examples[current_path].append(f"Object with {len(value)} keys")
                
                # Recursive gezinme
                if isinstance(value, dict):
                    self.traverse_json(value, current_path, False, max_samples, current_path)
                elif isinstance(value, list) and len(value) > 0:
                    # Array'in birden fazla elemanını analiz et
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
            # Ana seviyede array varsa
            if len(obj) > 0:
                items_to_check = min(len(obj), max_samples)
                for i in range(items_to_check):
                    item = obj[i]
                    if isinstance(item, dict):
                        # Root level array için parent_path "root" olmalı
                        self.traverse_json(item, path if path != "root" else "root", True, max_samples, "root")
                    else:
                        item_type = self.analyze_value(item)
                        self.field_types[path].add(item_type)
                        if len(self.field_examples[path]) < 3:
                            self.field_examples[path].append(item)
    
    def generate_markdown(self, title: str = "JSON Dokümantasyonu") -> str:
        """Markdown formatında dokümantasyon üretir"""
        
        md = f"# {title}\n\n"
        md += f"**Oluşturulma Tarihi:** {self._get_current_date()}\n\n"
        md += "---\n\n"
        
        md += "## İçindekiler\n\n"
        md += "1. [Genel Bakış](#genel-bakış)\n"
        md += "2. [Alan Detayları](#alan-detayları)\n"
        md += "3. [Veri Yapısı](#veri-yapısı)\n\n"
        
        md += "---\n\n"
        
        # Genel Bakış
        md += "## Genel Bakış\n\n"
        md += f"**Toplam Alan Sayısı:** {len(self.field_types)}\n\n"
        
        type_counts = defaultdict(int)
        for types in self.field_types.values():
            for t in types:
                type_counts[t] += 1
        
        md += "**Tip Dağılımı:**\n\n"
        for tip, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            md += f"- {tip}: {count} alan\n"
        
        md += "\n---\n\n"
        
        # Alan Detayları
        md += "## Alan Detayları\n\n"
        
        # Path'leri sırala
        sorted_paths = sorted(self.field_types.keys())
        
        for path in sorted_paths:
            types = self.field_types[path]
            examples = self.field_examples.get(path, [])
            frequency = self.field_frequency.get(path, 0)
            
            # Path seviyesine göre başlık boyutu
            level = path.count('.') + path.count('[]')
            header_level = min(level + 3, 6)
            
            md += f"{'#' * header_level} `{path}`\n\n"
            
            # Tip bilgisi
            type_str = " | ".join(sorted(types))
            md += f"**Tip:** {type_str}\n\n"
            
            # Opsiyonel/Zorunlu bilgisi
            # Root level için "root" parent'ını kontrol et
            parent_path = ""
            if '.' in path:
                parent_path = ".".join(path.split('.')[:-1])
            elif '[]' not in path:  # Root level alanlar
                parent_path = "root"
            
            if parent_path in self.total_objects_seen:
                total = self.total_objects_seen[parent_path]
                if frequency == total:
                    md += f"**Durum:** ✅ Zorunlu (tüm {total} örnekte var)\n\n"
                else:
                    percentage = (frequency / total * 100) if total > 0 else 0
                    md += f"**Durum:** ⚠️ Opsiyonel ({frequency}/{total} örnekte var, %{percentage:.1f})\n\n"
            
            # Örnek değerler
            if examples:
                md += "**Örnek Değerler:**\n\n"
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
        """Tek bir alan için schema oluşturur"""
        types = self.field_types.get(path, set())
        examples = self.field_examples.get(path, [])
        
        field_schema = {}
        
        # Tip belirleme
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
            # Birden fazla tip varsa anyOf kullan
            field_schema["anyOf"] = [{"type": t} for t in json_types]
        
        # Format bilgisi ekle
        for t in types:
            if "URL" in t:
                field_schema["format"] = "uri"
            elif "email" in t:
                field_schema["format"] = "email"
            elif "date" in t:
                field_schema["format"] = "date"
        
        # Array ise items bilgisini ekle
        if "array" in json_types:
            array_item_path = f"{path}[]"
            
            # Array'in alt alanlarını kontrol et (object içeriyor mu?)
            array_children = [p for p in self.field_types.keys() if p.startswith(f"{path}[].")]
            
            if array_children:
                # Array içinde object var
                field_schema["items"] = {
                    "type": "object",
                    "properties": {}
                }
                # Array içindeki object'in alt alanlarını ekle
                for sub_path in array_children:
                    # Array items path'lerini atla ([]' ile bitenleri)
                    if sub_path.endswith('[]'):
                        continue
                    if sub_path.count('.') == path.count('.') + 1:
                        field_name = sub_path.split('.')[-1]
                        field_schema["items"]["properties"][field_name] = self._build_field_schema(sub_path)
            elif array_item_path in self.field_types:
                # Array içinde primitive type var
                item_types = self.field_types[array_item_path]
                item_json_types = [type_mapping.get(t, "string") for t in item_types if t in type_mapping]
                if len(item_json_types) == 1:
                    field_schema["items"] = {"type": item_json_types[0]}
                elif len(item_json_types) > 1:
                    field_schema["items"] = {"anyOf": [{"type": t} for t in item_json_types]}
        
        # Object ise properties bilgisini ekle
        if "object" in json_types:
            # Bu object'in alt alanlarını bul
            child_paths = [p for p in self.field_types.keys() if p.startswith(f"{path}.") and p != path]
            
            if child_paths:
                # İmmediate children = sadece bir seviye derine olan path'ler
                # Bunu bulmak için: child_paths'leri incele
                immediate_children = []
                for p in child_paths:
                    # path'ten sonraki kısmı al
                    suffix = p[len(path) + 1:]  # +1 for the dot
                    
                    # Suffix'te  '[]' varsa kontrol et
                    if '[]' in suffix:
                        # suffix '[]' ile mi başlıyor?
                        if suffix == '[]' or suffix.startswith('[].'):
                            continue  # Bu immediate değil, array elemanı
                        # Suffix format: 'key[]' veya 'key[].something' olabilir
                        # 'key[]' immediate, 'key[].something' değil
                        bracket_pos = suffix.index('[]')
                        before_bracket = suffix[:bracket_pos]
                        after_bracket = suffix[bracket_pos + 2:]
                        
                        if '.' not in before_bracket and after_bracket == '':
                            immediate_children.append(p)
                    else:
                        # '[]' yok - normal key
                        # Deeper path kontrolü yap: sadece '.' ile devam eden path'leri say
                        # '[]' ile devam edenler aynı seviyede
                        has_deeper = any(cp.startswith(p + '.') and not cp.startswith(p + '[]')
                                       for cp in child_paths if cp != p)
                        if not has_deeper:
                            immediate_children.append(p)
                
                if immediate_children:
                    # Array items path'lerini filtrele ([]' ile bitenleri çıkar)
                    # Örnek: 'details[]' property değil, 'details' array'inin items'ı
                    immediate_children = [p for p in immediate_children if not p.endswith('[]')]
                    
                if immediate_children:
                    # Tüm key'leri al
                    all_keys = [p[len(path) + 1:] for p in immediate_children]
                    # '[]' içeren key'leri temizle (sadece key kısmını al)
                    all_keys = [k.split('[]')[0] if '[]' in k else k for k in all_keys]
                    unique_keys = set(all_keys)
                    
                    # Eğer key'lerde dosya uzantısı varsa (.docx, .pdf vb.) veya
                    # birçok farklı key varsa dinamik olabilir
                    file_extensions = ['.docx', '.pdf', '.xlsx', '.txt', '.json', '.xml', '.html']
                    looks_dynamic = any(any(k.endswith(ext) for ext in file_extensions) for k in unique_keys)
                    
                    if looks_dynamic and len(immediate_children) > 0:
                        # Dinamik key pattern - representative bir path al
                        representative_path = immediate_children[0]
                        field_schema["additionalProperties"] = self._build_field_schema(representative_path)
                    else:
                        # Sabit key'ler - normal properties
                        field_schema["properties"] = {}
                        for sub_path in immediate_children:
                            # Key ismini çıkar
                            field_name = sub_path[len(path) + 1:]
                            if '[]' in field_name:
                                field_name = field_name.split('[]')[0]
                            if field_name not in field_schema["properties"]:
                                field_schema["properties"][field_name] = self._build_field_schema(sub_path)
            else:
                # Alt alan yok, boş properties
                field_schema["properties"] = {}
        
        # Örnek değerleri ekle
        if examples:
            # String olan ve "Array" veya "Object" ile başlamayan örnekleri filtrele
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
    
    def generate_html(self, title: str = "JSON Dokümantasyonu") -> str:
        """HTML formatında dokümantasyon üretir"""
        
        html = f"""<!DOCTYPE html>
<html lang="tr">
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
        <p><strong>Oluşturulma Tarihi:</strong> {self._get_current_date()}</p>
        
        <div class="stats">
            <h2>Genel Bakış</h2>
            <p><strong>Toplam Alan Sayısı:</strong> {len(self.field_types)}</p>
        </div>
        
        <h2>Alan Detayları</h2>
"""
        
        sorted_paths = sorted(self.field_types.keys())
        
        for path in sorted_paths:
            types = self.field_types[path]
            examples = self.field_examples.get(path, [])
            
            type_str = " | ".join(sorted(types))
            
            html += f"""
        <div class="field">
            <div class="field-path">{path}</div>
            <p><span class="field-type">Tip:</span> {type_str}</p>
"""
            
            if examples:
                html += "            <p><strong>Örnek Değerler:</strong></p>\n"
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
        """HTML karakterlerini escape eder"""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    def _get_current_date(self) -> str:
        """Şu anki tarihi döndürür"""
        from datetime import datetime
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python json_documenter.py <json_dosyası> [--format md|html|schema|both|all]")
        print("\nÖrnek:")
        print("  python json_documenter.py data.json")
        print("  python json_documenter.py data.json --format html")
        print("  python json_documenter.py data.json --format schema")
        print("  python json_documenter.py data.json --format both     # md + html")
        print("  python json_documenter.py data.json --format all      # md + html + schema")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_format = "md"  # default
    
    if len(sys.argv) > 2 and sys.argv[2] == "--format":
        if len(sys.argv) > 3:
            output_format = sys.argv[3].lower()
    
    # JSON dosyasını oku
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"HATA: JSON dosyası okunamadı: {e}")
        sys.exit(1)
    
    # Analiz et
    print("JSON dosyası analiz ediliyor...")
    documenter = JSONDocumenter()
    documenter.traverse_json(data)
    
    # Dokümantasyon üret
    base_name = json_file.rsplit('.', 1)[0]
    
    if output_format in ["md", "both", "all"]:
        md_content = documenter.generate_markdown()
        md_file = f"{base_name}_documentation.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Markdown dokümantasyonu oluşturuldu: {md_file}")
    
    if output_format in ["html", "both", "all"]:
        html_content = documenter.generate_html()
        html_file = f"{base_name}_documentation.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ HTML dokümantasyonu oluşturuldu: {html_file}")
    
    if output_format in ["schema", "all"]:
        schema_content = documenter.generate_json_schema()
        schema_file = f"{base_name}_schema.json"
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write(schema_content)
        print(f"✓ JSON Schema oluşturuldu: {schema_file}")
    
    print(f"\nToplam {len(documenter.field_types)} alan analiz edildi.")


if __name__ == "__main__":
    main()