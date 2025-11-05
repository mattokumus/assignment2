#!/usr/bin/env python3
"""
ECHR Cases Data Extraction Script (FIXED VERSION)
Extracts: Country, Article, Year, Applicant Type, Outcome (Violation)

IMPORTANT: Only includes cases with SUBSTANTIVE decisions (violation/no-violation)
Excludes: procedural/jurisdictional decisions (inadmissible, struck out, etc.)
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path


# Define substantive conclusion types
SUBSTANTIVE_TYPES = {'violation', 'no-violation'}


def parse_date_to_year(date_string):
    """Parse date string to extract year"""
    if not date_string:
        return None
    try:
        # Format: "18/07/2017 00:00:00"
        date_obj = datetime.strptime(date_string, "%d/%m/%Y %H:%M:%S")
        return date_obj.year
    except:
        return None


def extract_applicant_type(docname, parties):
    """
    Extract applicant type from document name or parties
    Returns: 'Individual', 'Organization', 'Political Party', etc.
    """
    if not docname:
        return "Unknown"

    docname_lower = docname.lower()

    # Check for organizations/parties
    if "party" in docname_lower or "parties" in docname_lower:
        return "Political Party"
    elif "association" in docname_lower or "foundation" in docname_lower:
        return "Organization"
    elif "company" in docname_lower or "ltd" in docname_lower or "corporation" in docname_lower:
        return "Company"
    elif "and others" in docname_lower:
        return "Multiple Applicants"
    else:
        return "Individual"


def extract_judge_info(decision_body):
    """
    Extract judge information from decision_body array
    Returns: dict with judge names, president, and count

    decision_body format:
    [
        {"name": "Helena Jäderblom", "role": "president"},
        {"name": "Branko Lubarda", "role": "judges"},
        {"name": "Helen Keller", "role": "judges"}
    ]
    """
    if not decision_body or not isinstance(decision_body, list):
        return {
            'judges_all': '',
            'judge_president': '',
            'judge_count': 0,
            'judge_names_list': ''
        }

    president = ''
    all_judges = []

    for member in decision_body:
        name = member.get('name', '')
        role = member.get('role', '').lower()

        if name:
            all_judges.append(name)

            if 'president' in role:
                president = name

    return {
        'judges_all': '; '.join(all_judges),  # All judges separated by semicolon
        'judge_president': president,
        'judge_count': len(all_judges),
        'judge_names_list': '|'.join(all_judges)  # Alternative format with pipe separator
    }


def extract_violations(conclusion_list):
    """
    Extract violation information from conclusion array
    Returns: dict with violation info, or None if no substantive decision
    
    FIXED: Now properly distinguishes between:
    - Substantive decisions (violation/no-violation)
    - Procedural decisions (inadmissible, struck out, etc.)
    """
    if not conclusion_list:
        return None
    
    violation_articles = []
    no_violation_articles = []
    has_substantive = False
    procedural_types = []
    
    for conclusion in conclusion_list:
        article = conclusion.get('article', '')
        conclusion_type = conclusion.get('type', '')
        
        # Check if this is a substantive decision
        if conclusion_type in SUBSTANTIVE_TYPES:
            has_substantive = True
            
            if conclusion_type == 'violation':
                violation_articles.append(article)
            elif conclusion_type == 'no-violation':
                no_violation_articles.append(article)
        else:
            # Track procedural/other types
            if conclusion_type:
                procedural_types.append(conclusion_type)
    
    # If no substantive conclusion, return None (will be excluded)
    if not has_substantive:
        return None
    
    return {
        'has_violation': len(violation_articles) > 0,
        'violation_count': len(violation_articles),
        'no_violation_count': len(no_violation_articles),
        'violated_articles': violation_articles,
        'no_violation_articles': no_violation_articles,
        'has_procedural': len(procedural_types) > 0,
        'procedural_types': procedural_types
    }


def extract_case_data(case):
    """Extract relevant data from a single case"""

    # Country
    country_name = case.get('country', {}).get('name', '')
    country_code = case.get('country', {}).get('alpha2', '')

    # Article(s)
    articles = case.get('article', [])

    # Year
    judgement_date = case.get('judgementdate', '')
    year = parse_date_to_year(judgement_date)

    # Applicant Type
    docname = case.get('docname', '')
    parties = case.get('parties', [])
    applicant_type = extract_applicant_type(docname, parties)

    # Outcome (Violation) - FIXED VERSION
    conclusion = case.get('conclusion', [])
    violation_info = extract_violations(conclusion)

    # If no substantive decision, return None
    if violation_info is None:
        return None

    # Judge Information - NEW
    decision_body = case.get('decision_body', [])
    judge_info = extract_judge_info(decision_body)

    return {
        'itemid': case.get('itemid', ''),
        'appno': case.get('appno', ''),
        'docname': docname,
        'country_name': country_name,
        'country_code': country_code,
        'articles': ', '.join(articles) if articles else '',
        'year': year,
        'judgement_date': judgement_date,
        'applicant_type': applicant_type,
        'has_violation': violation_info['has_violation'],
        'violation_count': violation_info['violation_count'],
        'no_violation_count': violation_info['no_violation_count'],
        'violated_articles': ', '.join(violation_info['violated_articles']),
        'no_violation_articles': ', '.join(violation_info['no_violation_articles']),
        'has_mixed_decision': violation_info['has_procedural'],  # Has both substantive + procedural
        # Judge information
        'judge_president': judge_info['judge_president'],
        'judge_count': judge_info['judge_count'],
        'judges_all': judge_info['judges_all'],
        'judge_names_list': judge_info['judge_names_list']
    }


def process_json_file(input_file, output_file='extracted_data.csv'):
    """Process JSON file and extract data to CSV"""
    
    print(f"Reading JSON file: {input_file}")
    
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data is a list or single object
    if isinstance(data, dict):
        cases = [data]
    elif isinstance(data, list):
        cases = data
    else:
        raise ValueError("JSON data must be a dict or list")
    
    print(f"Processing {len(cases)} cases...")
    
    # Extract data from each case
    extracted_data = []
    excluded_count = 0
    
    for i, case in enumerate(cases, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(cases)} cases...")
        
        try:
            case_data = extract_case_data(case)
            
            if case_data is None:
                # Case has no substantive decision - exclude it
                excluded_count += 1
            else:
                extracted_data.append(case_data)
                
        except Exception as e:
            print(f"  Error processing case {i}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(extracted_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n{'='*80}")
    print("✓ DATA EXTRACTION COMPLETED")
    print(f"{'='*80}")
    print(f"\n📊 SUMMARY:")
    print(f"  • Total cases in file: {len(cases)}")
    print(f"  • Cases with SUBSTANTIVE decisions: {len(df)} ({len(df)/len(cases)*100:.1f}%)")
    print(f"  • Cases EXCLUDED (procedural only): {excluded_count} ({excluded_count/len(cases)*100:.1f}%)")
    print(f"  • Saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (SUBSTANTIVE CASES ONLY)")
    print(f"{'='*80}")
    print(f"  Countries: {df['country_name'].nunique()}")
    print(f"  Years: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"  Cases with violations: {df['has_violation'].sum()} ({df['has_violation'].sum()/len(df)*100:.1f}%)")
    print(f"  Cases without violations: {(~df['has_violation']).sum()} ({(~df['has_violation']).sum()/len(df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("TOP 10 COUNTRIES (SUBSTANTIVE CASES)")
    print(f"{'='*80}")
    print(df['country_name'].value_counts().head(10))
    
    print(f"\n{'='*80}")
    print("APPLICANT TYPES (SUBSTANTIVE CASES)")
    print(f"{'='*80}")
    print(df['applicant_type'].value_counts())

    print(f"\n{'='*80}")
    print("JUDGE INFORMATION")
    print(f"{'='*80}")
    print(f"  • Cases with judge information: {df['judge_count'].gt(0).sum()} ({df['judge_count'].gt(0).sum()/len(df)*100:.1f}%)")
    print(f"  • Unique judges: {len(set([j for judges in df['judge_names_list'].dropna() for j in judges.split('|') if j]))}")
    print(f"  • Average judges per case: {df['judge_count'].mean():.1f}")
    print(f"  • Most common presidents:")
    print(df['judge_president'][df['judge_president'] != ''].value_counts().head(5))

    print(f"\n{'='*80}")
    print("⚠️  IMPORTANT NOTE:")
    print(f"{'='*80}")
    print("""
This dataset NOW includes ONLY cases with substantive decisions:
  ✓ Cases with "violation" conclusions
  ✓ Cases with "no-violation" conclusions
  
EXCLUDED cases (procedural/jurisdictional):
  ✗ Inadmissible
  ✗ Struck out
  ✗ Lack of jurisdiction
  ✗ Preliminary objections
  ✗ Other procedural outcomes
  
This ensures clean analysis focused on the research question:
"Does the ECtHR treat countries differently in substantive decisions?"
    """)
    
    return df


def main():
    """Main function"""
    
    # Fixed filenames
    input_file = 'cases-2000.json'
    output_file = 'extracted_data.csv'
    
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found!")
        print(f"Please make sure '{input_file}' is in the same folder as this script.")
        return
    
    # Process the file
    df = process_json_file(input_file, output_file)
    
    print(f"\n✓ Done! Check '{output_file}' for results.")


if __name__ == "__main__":
    main()