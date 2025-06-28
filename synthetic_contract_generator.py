"""
Enhanced Synthetic Construction Contract Data Generator
=====================================================

A comprehensive system for generating synthetic construction contracts with:
- Real-time data validation and anonymization
- Interactive Streamlit interface
- User-defined contract sections
- LLM-powered content generation with fallback mechanisms
- Privacy-preserving synthetic data generation

Author: MERVIN JOSEPH L
Date: June 2025
Version: 1.0.0
"""

import os
import re
import json 
import logging
import random
import streamlit as st
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import google.generativeai as genai
from faker import Faker
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_error_location(cls_instance=None) -> str:
    """Dynamically gets the class and function name for error logging."""
    try:
        frame = sys._getframe(1) 
        func_name = frame.f_code.co_name
        if cls_instance:
            class_name = cls_instance.__class__.__name__
            return f"{class_name}.{func_name}"
        return func_name
    except Exception: 
        return "UnknownLocation"

class ContractType(Enum):
    """Enumeration of construction contract types with descriptions."""
    LUMP_SUM = "Lump Sum - Fixed price for entire project"
    TIME_MATERIALS = "Time & Materials - Payment based on actual time and materials"
    COST_PLUS = "Cost Plus - Actual costs plus agreed fee"
    UNIT_PRICE = "Unit Price - Payment per unit of work completed"

class ProjectComplexity(Enum):
    """Enumeration of project complexity levels affecting contract terms."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MEGA = "mega"

class ContractArticle(Enum):
    """Available contract articles that can be selected for generation."""
    DEFINITIONS = "Definitions and Interpretation"
    SCOPE_OF_WORK = "Scope of Work and Specifications"
    CONTRACT_PRICE = "Contract Price and Payment Terms"
    TIME_PERFORMANCE = "Time for Performance and Delays"
    CHANGE_ORDERS = "Change Orders and Modifications"
    QUALITY_CONTROL = "Quality Control and Inspections"
    INSURANCE_BONDING = "Insurance and Bonding Requirements"
    SAFETY_COMPLIANCE = "Safety and Environmental Compliance"
    DEFAULT_TERMINATION = "Default and Termination"
    DISPUTE_RESOLUTION = "Dispute Resolution"
    GENERAL_PROVISIONS = "General Provisions"
    SIGNATURE_BLOCKS = "Signature Blocks"

@dataclass
class ContractRequirements:
    """Data structure for contract generation requirements."""
    project_type: str
    contract_value: int
    duration_days: int
    location: str
    complexity: str
    selected_articles: List[str] = field(default_factory=list)
    special_requirements: List[str] = field(default_factory=list)
    regulatory_requirements: List[str] = field(default_factory=list) # Added for completeness
    risk_factors: List[str] = field(default_factory=list) # Added for completeness

@dataclass
class GeneratedContractData:
    """Comprehensive data structure for generated contract information."""
    contract_date: str
    project_name: str
    project_description: str
    owner_name: str
    owner_representative: str
    contractor_name: str
    contractor_address: str
    contractor_representative: str
    contract_amount: str
    contract_type: str
    payment_terms: str
    retainage_percentage: str
    start_date: str
    completion_date: str
    project_duration: str
    governing_law: str
    dispute_resolution: str
    insurance_requirements: List[str] = field(default_factory=list)
    bond_requirements: List[str] = field(default_factory=list)
    building_codes: List[str] = field(default_factory=list)
    quality_standards: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)

class SyntheticDataGenerator:
    """Generates realistic but synthetic data for construction contracts."""

    # This function is to generate a random data using faker library
    def __init__(self, seed: Optional[int] = None):
        try:
            if seed is None:
                seed = random.randint(1, 10000)
            self.fake = Faker()
            Faker.seed(seed)
            random.seed(seed)
            self.project_types = [
                "Highway Construction and Reconstruction", "Bridge Design and Construction",
                "Municipal Water Treatment Facility", "Educational Institution Building",
                "Public Safety Complex", "Airport Infrastructure Development",
                "Transit System Expansion", "Environmental Remediation Project",
                "Energy Infrastructure Installation", "Telecommunications Network Deployment",
                "Residential Development Complex", "Commercial Office Building",
                "Industrial Manufacturing Facility"
            ]
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # This function generates a full set of contract data including project details, dates, parties, and compliance requirements based on the given contract inputs.
    def generate_contract_data(self, requirements: ContractRequirements) -> GeneratedContractData:
        try:
            contract_date = self.fake.date_between(start_date='-3m', end_date='+1m')
            start_date = contract_date + timedelta(days=random.randint(15, 60))
            completion_date = start_date + timedelta(days=requirements.duration_days)
            contractor_types = ["Construction", "Builders", "Engineering", "Development", "Infrastructure"]
            contractor_entities = ["LLC", "Inc.", "Corp.", "Co."]

            return GeneratedContractData(
                contract_date=contract_date.strftime("%B %d, %Y"),
                project_name=f"{requirements.project_type} - Phase {random.randint(1,5)}",
                project_description=f"Comprehensive {requirements.project_type.lower()} project including all associated work, materials, installation, and completion for the site located at {requirements.location}.",
                owner_name=f"The {self.fake.word().capitalize()} Authority of {requirements.location.split(' ')[0]}",
                owner_representative=f"[OWNER_REPRESENTATIVE_NAME], P.Eng., Director of Capital Projects",
                contractor_name=f"{self.fake.bs().split(' ')[0].capitalize()} {random.choice(contractor_types)} {random.choice(contractor_entities)}",
                contractor_address=f"[CONTRACTOR_STREET_NUMBER] [CONTRACTOR_STREET_NAME], [CONTRACTOR_CITY], [CONTRACTOR_STATE_PROVINCE] [CONTRACTOR_POSTAL_CODE]",
                contractor_representative=f"[CONTRACTOR_REPRESENTATIVE_FIRST_NAME] [CONTRACTOR_REPRESENTATIVE_LAST_NAME], Senior Project Manager",
                contract_amount=f"${requirements.contract_value:,.2f}",
                contract_type=random.choice(list(ContractType)).value,
                payment_terms="Net 30 days upon submission and Owner's approval of itemized invoice, subject to retainage provisions.",
                retainage_percentage=f"{random.choice([5.0, 7.5, 10.0]):.1f}%",
                start_date=start_date.strftime("%B %d, %Y"),
                completion_date=completion_date.strftime("%B %d, %Y"),
                project_duration=f"{requirements.duration_days} calendar days from the official Notice to Proceed date.",
                governing_law="The laws of the State of [JURISDICTION_STATE_PROVINCE], without regard to its conflict of law principles.",
                dispute_resolution="Binding arbitration administered by the [ARBITRATION_ADMINISTRATOR_NAME] in accordance with its Construction Industry Arbitration Rules, with the arbitration taking place in [ARBITRATION_CITY], [ARBITRATION_STATE_PROVINCE].",
                insurance_requirements=self._generate_insurance_requirements(requirements.contract_value),
                bond_requirements=self._generate_bond_requirements(requirements.contract_value),
                building_codes=self._generate_building_codes(requirements.location),
                quality_standards=self._generate_quality_standards(),
                safety_requirements=self._generate_safety_requirements()
            )
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise
    
    # This Function returns the base requirements that are necessary for the contract based on the contract value
    def _generate_insurance_requirements(self, contract_value: int) -> List[str]:
        try:
            base_requirements = [
                "Commercial General Liability: $2,000,000 per occurrence / $4,000,000 aggregate",
                "Automobile Liability: $1,000,000 combined single limit (CSL)",
                "Workers' Compensation: Statutory limits as per [JURISDICTION_STATE_PROVINCE] law",
                "Employer's Liability: $1,000,000 each accident / $1,000,000 disease policy / $1,000,000 disease each employee"
            ]
            if contract_value > 1_000_000:
                base_requirements.append("Professional Liability (E&O): $2,000,000 per claim / $2,000,000 aggregate (if design services provided)")
            if contract_value > 5_000_000:
                base_requirements.extend([
                    "Umbrella/Excess Liability: $5,000,000 per occurrence / $5,000,000 aggregate",
                    "Pollution Liability: $2,000,000 per occurrence (if scope involves hazardous materials)"
                ])
            if contract_value > 20_000_000: # Mega projects
                 base_requirements[0] = "Commercial General Liability: $5,000,000 per occurrence / $10,000,000 aggregate"
                 idx_umbrella = next((i for i, s in enumerate(base_requirements) if "Umbrella/Excess Liability" in s), -1)
                 if idx_umbrella != -1:
                     base_requirements[idx_umbrella] = "Umbrella/Excess Liability: $10,000,000 per occurrence / $10,000,000 aggregate"
            return base_requirements
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # This Function returns the base requirements that are necessary for the creating the bond
    def _generate_bond_requirements(self, contract_value: int) -> List[str]:
        try:
            if contract_value < 100_000:
                return ["Bonding requirements may be waived by Owner for contract values under $100,000."]
            requirements = [
                "Performance Bond: 100% of the Contract Price, from an A.M. Best 'A-' rated surety.",
                "Payment Bond: 100% of the Contract Price, from an A.M. Best 'A-' rated surety."
            ]
            if contract_value > 1_000_000:
                requirements.append("Maintenance Bond: 10% of Contract Price for 24 months post-Substantial Completion.")
            elif contract_value > 250_000 :
                requirements.append("Maintenance Bond: 10% of Contract Price for 12 months post-Substantial Completion.")
            return requirements
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # This function returns a list of standard building codes applicable to the specified project location.
    def _generate_building_codes(self, location: str) -> List[str]:
        try:
            return [
                "International Building Code (IBC), latest edition adopted by [JURISDICTION_STATE_PROVINCE].",
                "International Fire Code (IFC), International Mechanical Code (IMC), International Plumbing Code (IPC) - latest adopted editions.",
                "National Electrical Code (NEC), latest adopted edition.",
                "All applicable local [JURISDICTION_CITY_NAME] and County of [JURISDICTION_COUNTY_NAME] building codes, ordinances, and amendments.",
                "Americans with Disabilities Act (ADA) Standards for Accessible Design, current version."
            ]
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise
    
    # This function returns a list of general quality standards and industry codes for construction materials and methods.
    def _generate_quality_standards(self) -> List[str]:
        try:
            return [
                "All materials and workmanship must conform to relevant ASTM International, ANSI, and other applicable industry standards.",
                "Structural steel fabrication and erection to comply with AISC (American Institute of Steel Construction) specifications.",
                "Concrete work must adhere to ACI (American Concrete Institute) codes and standards (e.g., ACI 318, ACI 301).",
                "Welding procedures and welders to be qualified in accordance with AWS (American Welding Society) D1.1/D1.1M or other applicable AWS codes.",
                "A comprehensive Quality Control (QC) Plan must be submitted by Contractor and approved by Owner prior to work commencement."
            ]
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # This function returns a list of mandatory safety requirements and protocols to be followed during construction.
    def _generate_safety_requirements(self) -> List[str]:
        try:
            return [
                "Full compliance with all applicable federal, state, and local safety regulations, including OSHA (Occupational Safety and Health Administration) standards (e.g., 29 CFR 1926).",
                "Contractor shall develop and implement a site-specific Health and Safety Plan (HASP), submitted for Owner review prior to mobilization.",
                "Daily pre-task safety briefings (toolbox talks) for all personnel on site.",
                "Mandatory use of appropriate Personal Protective Equipment (PPE) as required by task and site conditions.",
                "Clear GHS-compliant labeling for all hazardous materials, with Safety Data Sheets (SDS) readily available."
            ]
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

class ContentValidator:
    """Validates and anonymizes content for privacy protection."""

    # This constructor initializes regex patterns and replacement templates for detecting and organizational information.
    def __init__(self):

        try:
            self.sensitive_patterns = {
                'phone_numbers': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
                'ssn_numbers': r'\b\d{3}-\d{2}-\d{4}\b',
                'real_company_names': self._get_real_company_patterns(),
                'specific_addresses': r'\b\d{1,5}\s+(?:[A-Za-z0-9\s.-]+)\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Way|Terrace|Circle|Pkwy|Freeway|Hwy)\b',
                'license_numbers': r'\b[A-Z]{1,3}[-.\s]?\d{4,10}\b',
                'bank_routing': r'\b\d{9}\b',
                'credit_card_numbers': r'\b(?:\d[ -]*?){13,19}\b'
            }
            self.replacement_templates = {
                'phone_numbers': '[PHONE_NUMBER_REDACTED]', 'email_addresses': '[EMAIL_ADDRESS_REDACTED]',
                'ssn_numbers': '[SSN_REDACTED]', 'real_company_names': '[COMPANY_NAME_GENERICIZED]',
                'specific_addresses': '[ADDRESS_DETAIL_REDACTED]', 'license_numbers': '[LICENSE_NUMBER_REDACTED]',
                'bank_routing': '[BANK_ROUTING_REDACTED]', 'credit_card_numbers': '[CREDIT_CARD_REDACTED]'
            }
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # This function scans the input content for sensitive information, anonymizes it using predefined patterns, and returns the cleaned content along with any issues found.
    def validate_and_anonymize_content(self, content: str) -> Tuple[str, List[str]]:

        try:
            issues_found = []
            cleaned_content = content
            if not isinstance(cleaned_content, str):
                logger.warning(f"Content was not string (type: {type(cleaned_content)}), converting.")
                cleaned_content = str(cleaned_content)
                issues_found.append(f"Warning: Content converted to string. Original type: {type(content)}")

            for pattern_name, pattern_regex in self.sensitive_patterns.items():
                def replace_match(match_obj): return self.replacement_templates[pattern_name]
                cleaned_content, num_replacements = re.subn(pattern_regex, replace_match, cleaned_content, flags=re.IGNORECASE)
                if num_replacements > 0:
                    issues_found.append(f"Anonymized {num_replacements} instance(s) of potential {pattern_name}.")
            additional_issues = self._validate_construction_context(cleaned_content)
            issues_found.extend(additional_issues)
            return cleaned_content, issues_found
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            return content, [f"Error during validation: {str(e)}"]


    # This function returns a regex pattern string matching well-known real-world company names and legal entity suffixes for anonymization.
    def _get_real_company_patterns(self) -> str:
        try:
            real_company_indicators = [
                r'\b(?:Microsoft|Google|Apple|Amazon|Meta|Tesla|Ford|Boeing|Caterpillar|General\s+Electric|Bechtel|Skanska|Turner\s+Construction|Kiewit|Fluor\s+Corp|Jacobs|AECOM|Vinci|Acciona)\b',
                r'\b(?:Inc\.?|LLC\.?|Corp\.?|Ltd\.?|GmbH)\s*(?:,|\.|$)',
            ]
            return '|'.join(real_company_indicators)
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # This function scans the content for construction-specific sensitive references like brand names, patent numbers, and project codes, returning any issues found.
    def _validate_construction_context(self, content: str) -> List[str]:
        try:
            issues = []
            if re.search(r'Patent\s+(?:No\.?\s*)?\b[A-Z0-9]{7,12}\b', content, re.IGNORECASE):
                issues.append("Potential patent number reference found. Ensure generic or placeholder.")
            brand_patterns = r'\b(?:Caterpillar|CAT|John\s+Deere|Volvo\s+CE|Liebherr|Komatsu|Hitachi|Hilti|DeWalt|Makita|Bosch|Cummins|Perkins)\b'
            found_brands = re.findall(brand_patterns, content, re.IGNORECASE)
            if found_brands and not all(brand.startswith('[') and brand.endswith(']') for brand in found_brands):
                 issues.append(f"Specific equipment/material brand(s) found: {', '.join(set(found_brands))}. Consider genericizing (e.g., '[BRAND_EQUIVALENT]').")
            if re.search(r'\b[A-Z]{2,4}-\d{3,6}-[A-Z0-9]{2,5}(?:-[A-Z])?\b', content):
                issues.append("Potential specific project code or internal identifier found. Ensure generic.")
            return issues
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            return [f"Error during construction context validation: {str(e)}"]

class LLMContentGenerator:
    """Manages LLM interactions for content generation."""

    # Initializes the LLM interface with an optional API key and attempts model setup.
    def __init__(self, api_key: Optional[str] = None):
        try:
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            self.model = None
            self.is_initialized = False
            self._setup_model()
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # Configures and initializes the Gemini LLM model using the provided or environment API key.
    def _setup_model(self) -> bool:
        try:
            if not self.api_key:
                logger.warning("No API key for LLM. LLM generation unavailable.")
                if 'st' in sys.modules: st.sidebar.warning("No Google API Key. LLM features disabled.", icon="⚠️")
                return False
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.is_initialized = True
            logger.info("LLM model (gemini-1.5-flash-latest) initialized.")
            if 'st' in sys.modules: st.sidebar.success("Google LLM Initialized.", icon="✅")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}", exc_info=True)
            if 'st' in sys.modules: st.sidebar.error(f"LLM Init Failed: {e}", icon="❌")
            self.model = None; self.is_initialized = False
            return False

    # Generates contract article content using the LLM or falls back if the model is unavailable or fails.
    def generate_article_content(self, article_type: str, requirements: ContractRequirements,
                               contract_data: GeneratedContractData, max_retries: int = 2,
                               max_tokens_per_call: int = 4000) -> str:
        try:
            if not self.model or not self.is_initialized:
                logger.warning(f"LLM unavailable for '{article_type}', using fallback.")
                return self._fallback_generation(article_type, requirements, contract_data)

            initial_prompt = self._create_article_prompt(article_type, requirements, contract_data)
            generated_text_parts = []
            current_prompt = initial_prompt
            safety_settings = [{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
            if 'st' in sys.modules: st.write(f"LLM: Generating content for '{article_type}'...")

            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        current_prompt,
                        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens_per_call, temperature=0.65, top_p=0.95),
                        safety_settings=safety_settings
                    )
                    if not response.parts or not hasattr(response, 'text'):
                        logger.error(f"LLM response for '{article_type}' empty/invalid. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                            if 'st' in sys.modules: st.error(f"'{article_type}' blocked: {response.prompt_feedback.block_reason_message}")
                            return self._fallback_generation(article_type, requirements, contract_data)
                        if attempt < max_retries -1 :
                            if 'st' in sys.modules: st.write(f"LLM: Empty/invalid response for '{article_type}', retrying...")
                            continue
                        else: raise Exception("LLM returned empty/invalid response after multiple retries.")

                    new_content = response.text.strip()
                    generated_text_parts.append(new_content)
                    if 'st' in sys.modules: st.write(f"LLM: Received chunk for '{article_type}'.")
                    finish_reason = response.candidates[0].finish_reason.name if response.candidates and response.candidates[0].finish_reason else ""
                    if finish_reason == "MAX_TOKENS" and len(generated_text_parts) < 3:
                        if 'st' in sys.modules: st.write(f"LLM: Max tokens for '{article_type}', attempting to continue...")
                        current_prompt = f"{initial_prompt}\n\nPREVIOUSLY GENERATED (DO NOT REPEAT):\n{''.join(generated_text_parts)}\n\nCONTINUE WRITING SEAMLESSLY:"
                    else:
                        logger.info(f"LLM for '{article_type}' finished. Reason: {finish_reason or 'Completed'}")
                        break
                except Exception as e_inner:
                    logger.error(f"LLM generation for '{article_type}' attempt {attempt + 1} failed: {e_inner}", exc_info=True)
                    if 'st' in sys.modules: st.write(f"LLM Error (Attempt {attempt+1}) for '{article_type}': {e_inner}")
                    if attempt == max_retries - 1:
                        if 'st' in sys.modules: st.warning(f"LLM failed for '{article_type}' after {max_retries} attempts. Using fallback.", icon="⚠️")
                        return self._fallback_generation(article_type, requirements, contract_data)
            
            final_generated_text = "\n\n".join(generated_text_parts)
            if not final_generated_text.strip():
                logger.warning(f"LLM empty content for '{article_type}'. Using fallback.")
                if 'st' in sys.modules: st.warning(f"LLM produced empty content for '{article_type}'. Using fallback.", icon="⚠️")
                return self._fallback_generation(article_type, requirements, contract_data)
            if 'st' in sys.modules: st.write(f"LLM: Successfully generated content for '{article_type}'.")
            return final_generated_text
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)} for '{article_type}': {str(e)}", exc_info=True)
            return self._fallback_generation(article_type, requirements, contract_data)
        
    # Uses a predefined template-based fallback to generate article content when LLM fails.
    def _fallback_generation(self, article_type: str, requirements: ContractRequirements,
                           contract_data: GeneratedContractData) -> str:
        try:
            logger.info(f"Using fallback template for article: {article_type}")
            fallback_generator = FallbackGenerator()
            return fallback_generator.generate_fallback_article(article_type, requirements, contract_data)
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)} while calling FallbackGenerator: {str(e)}", exc_info=True)
            return f"ARTICLE - {article_type.upper()}\n\n[CRITICAL ERROR: Fallback generation mechanism itself failed for this article. Please report this issue. Details: {str(e)}]"
        
    # Constructs the LLM prompt string for generating a specific contract article based on context and requirements.
    def _create_article_prompt(self, article_type: str, requirements: ContractRequirements,
                             contract_data: GeneratedContractData) -> str:
        try:
            base_context = f"""
Role: Senior Paralegal specializing in Construction Contracts.
Task: Draft a specific article for a construction contract.
Project Context: Type: {requirements.project_type}, Value: {contract_data.contract_amount}, Duration: {contract_data.project_duration}, Complexity: {requirements.complexity}, Location: {requirements.location}.
Instructions:
1. Use ONLY generic placeholders (e.g., [OWNER_NAME], [CONTRACTOR_NAME], [PROJECT_START_DATE]). No real entities.
2. Structure with clear numbered/lettered sub-sections (1.1, 1.2, a, b).
3. Content must be comprehensive, professional, legally sound, and formal.
4. Focus SOLELY on the requested article's content. Do not add external intro/conclusions.
5. Ensure keywords relevant to construction and the specific article are used (e.g. "installation", "materials", "completion", "subcontractor", "payment", "default" etc. as appropriate).
"""
            article_specific_prompts = {
                ContractArticle.DEFINITIONS.value: f"""{base_context}
ARTICLE TASK: "Definitions and Interpretation".
Define: "Agreement", "Change Order", "Contract Documents", "Day", "Drawings", "Final Completion", "Hazardous Materials", "Owner", "Contractor", "Project", "Site", "Specifications", "Subcontractor", "Substantial Completion", "Work", "Notice to Proceed".
Interpretation subsection: order of precedence, singular/plural, headings, governing law (use "[JURISDICTION_STATE_PROVINCE]").""",
                ContractArticle.SCOPE_OF_WORK.value: f"""{base_context}
ARTICLE TASK: "Scope of Work and Specifications" for [CONTRACTOR_NAME] for [OWNER_NAME].
Sub-sections:
1. General Description: Construction of a new {requirements.project_type} at {requirements.location}.
2. Contractor's Responsibilities: Furnish all labor, materials, equipment, tools, supervision, permits (as allocated), quality control, safety compliance, site cleanup, and coordination for full execution and completion of the Work.
3. Work Included: List major components for a {requirements.project_type} of {requirements.complexity} complexity (e.g., site preparation, foundations, structural framing, building envelope, interior fit-out, MEP systems installation, testing and commissioning, final cleanup and demobilization).
4. Specifications and Standards: Adherence to Contract Documents ([DRAWING_SET_IDENTIFIER], [SPECIFICATIONS_MANUAL_VERSION]), building codes ({', '.join(contract_data.building_codes)}), quality standards ({', '.join(contract_data.quality_standards)}), and safety requirements ({', '.join(contract_data.safety_requirements)}).
5. Exclusions (Optional but good): List any significant items NOT part of Contractor's scope if it clarifies boundaries.
Special Project Requirements: {', '.join(requirements.special_requirements) if requirements.special_requirements else 'Standard industry practices for a project of this nature.'}""",
                ContractArticle.CONTRACT_PRICE.value: f"""{base_context}
ARTICLE TASK: "Contract Price and Payment Terms".
Sub-sections:
1. Contract Price: Total sum {contract_data.contract_amount} ([CONTRACT_AMOUNT_IN_WORDS_PLACEHOLDER]). Basis: {contract_data.contract_type}. Inclusions: All labor, materials, equipment, overhead, profit.
2. Schedule of Values: Contractor to submit detailed SOV for Owner approval before first payment application.
3. Applications for Payment: Monthly by [DAY_OF_MONTH_FOR_SUBMISSION], detailing work completed, stored materials. Required backup: Lien waivers, payroll reports ([PAYROLL_FORM_ID_PLACEHOLDER]).
4. Review and Approval: Owner/Architect review within [NUMBER_OF_DAYS_FOR_REVIEW] days.
5. Payments: Net [NUMBER_OF_DAYS_FOR_PAYMENT_AFTER_APPROVAL] days from certification.
6. Retainage: {contract_data.retainage_percentage} withheld from each progress payment. Release upon Final Completion and satisfaction of all closeout requirements.
7. Final Payment: Conditions: Completion of all punch list items, submission of as-built drawings, warranties, final lien waivers, consent of surety.
""",
            }
            prompt = article_specific_prompts.get(article_type)
            if prompt: return prompt
            logger.info(f"Using generic LLM prompt for article: {article_type}")
            return f"{base_context}\n\nARTICLE TASK: Generate a comprehensive \"{article_type}\" article. Include all typical sub-clauses, provisions, and legal language. Well-structured."
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    # Formats a list of strings into a bulleted prompt-ready structure, with optional description labels.
    def _format_list_for_prompt(self, items: List[str], item_description: str) -> str:
        try:
            if not items: return f"    - No specific {item_description.lower()} listed; state general requirements."
            return "\n".join([f"    - {item_description}: {req}" for req in items])
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

class FallbackGenerator:
    """Provides fallback contract generation using templates."""
    #Initializes the FallbackGenerator instance and loads article templates, raising an error if initialization fails.
    def __init__(self):
        try:
            self.article_templates = self._initialize_templates()
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise
    
    #Generates a fallback article using the corresponding template and contract data, with automatic fallback to a generic version if specific formatting fails.
    def generate_fallback_article(self, article_type: str, requirements: ContractRequirements,
                                contract_data: GeneratedContractData) -> str:
        try:
            template = self.article_templates.get(article_type, self._get_generic_template(article_type))
            logger.info(f"Using fallback template for article: {article_type}")
            placeholder_data = {
                **asdict(contract_data), **asdict(requirements),
                'liquidated_damages_per_day': f"${max(500, int(requirements.contract_value * 0.0001) if requirements.contract_value > 0 else 500):,.0f}",
                'insurance_requirements_list': "\n".join([f"  - {item}" for item in contract_data.insurance_requirements]) or "  - Standard industry insurance as per [INSURANCE_SCHEDULE_REF].",
                'bond_requirements_list': "\n".join([f"  - {item}" for item in contract_data.bond_requirements]) or "  - Standard industry bonding as per [BONDING_SCHEDULE_REF].",
                'building_codes_list': "\n".join([f"  - {item}" for item in contract_data.building_codes]) or "  - Applicable building codes per [JURISDICTION_CODES_REF].",
                'quality_standards_list': "\n".join([f"  - {item}" for item in contract_data.quality_standards]) or "  - Adherence to relevant quality standards (ASTM, ACI, AISC).",
                'safety_requirements_list': "\n".join([f"  - {item}" for item in contract_data.safety_requirements]) or "  - Compliance with OSHA and site safety plan [SAFETY_PLAN_REF]."
            }
            try:
                class SafeDict(dict):
                    def __missing__(self, key):
                        return f"[{key}]"
                customized_content = template.format_map(SafeDict(placeholder_data))
            except Exception as e: 
                logger.error(f" There Might be an exception in the code {e}.", exc_info=True)
            return customized_content
        except KeyError as e_key:
            logger.error(f"Missing key in fallback template for {article_type}: {e_key}. Template might not be fully populated with available data.", exc_info=True)
            try:
                generic_template_content = self._get_generic_template(article_type).format_map(placeholder_data)
                return generic_template_content + f"\n\n[Note: Specific template for '{article_type}' had missing key: {e_key}. Displaying generic content.]"
            except Exception as e_generic_format:
                 logger.error(f"Failed to format even generic template for {article_type} after key error: {e_generic_format}", exc_info=True)
                 return f"ARTICLE {article_type.upper()}\n\n[Fallback content generation failed due to template key error '{e_key}' and subsequent generic template formatting error. Please check template definitions and available data.]"
        except Exception as e:
            logger.error(f"Error customizing fallback template for {article_type}: {e}", exc_info=True)
            return f"ARTICLE {article_type.upper()}\n\n[Fallback content generation for this article encountered an error: {str(e)}. Standard provisions apply as per industry practice.]"

    #Loads and returns a dictionary of predefined fallback templates for various contract article types.
    def _initialize_templates(self) -> Dict[str, str]:
        try:
            return {
                ContractArticle.DEFINITIONS.value: """ARTICLE - DEFINITIONS AND INTERPRETATION
1.1 Definitions: Key terms used herein are defined as follows:
  a) "Agreement": This Construction Contract, attachments, and amendments.
  b) "Contractor": {contractor_name}, the entity responsible for performing the Work.
  c) "Owner": {owner_name}, the entity commissioning the Project.
  d) "Project": The {project_type} located at {location}, detailed in Contract Documents.
  e) "Work": All labor, materials, equipment, installation, and services required by Contract Documents.
  f) "Contract Price": The total sum of {contract_amount}, payable to Contractor.
  g) "Substantial Completion": Stage where Project is usable for its intended purpose by Owner.
  h) "Final Completion": Stage where all Work, including punch list items, is complete and accepted.
1.2 Interpretation:
  a) Headings are for convenience only, not for interpretation.
  b) Singular terms include plural, and vice-versa, as context requires.
  c) This Agreement is governed by laws of {governing_law}.""",
                ContractArticle.SCOPE_OF_WORK.value: """ARTICLE - SCOPE OF WORK AND SPECIFICATIONS
1.1 General Scope: Contractor shall provide all labor, materials, equipment, tools, construction aids, supervision, and services necessary for the complete and proper execution of the Work described in the Contract Documents for the {project_type} ("Project") at {location}. The Project complexity is {complexity}.
1.2 Description of Work: The Work includes, but is not limited to:
  (List key phases appropriate for {project_type} and {complexity}, e.g.:
  a) Site preparation, demolition (if any), and earthwork.
  b) Foundation construction and structural framing.
  c) Building envelope (roofing, walls, windows, doors).
  d) Interior fit-out, finishes, and MEP (Mechanical, Electrical, Plumbing) systems installation.
  e) Testing, commissioning, and startup of all systems.
  f) Final site cleanup, landscaping (if applicable), and demobilization.)
1.3 Contractor's Responsibilities:
  a) Perform Work in a good, workmanlike, and expeditious manner.
  b) Comply with all applicable laws, codes, permits, and regulations.
  c) Coordinate with Owner, Architect/Engineer ([ARCHITECT_ENGINEER_NAME_PLACEHOLDER]), and Subcontractors.
  d) Implement and maintain quality control and safety programs.
1.4 Specifications and Standards: All Work shall conform to:
{building_codes_list}
{quality_standards_list}
{safety_requirements_list}
And all drawings and specifications listed in [DRAWING_SCHEDULE_REF] and [SPECIFICATIONS_INDEX_REF].""",
                ContractArticle.CONTRACT_PRICE.value: """ARTICLE - CONTRACT PRICE AND PAYMENT TERMS
2.1 Contract Price: Owner shall pay Contractor for full performance of the Work the sum of {contract_amount} (the "Contract Price"). This is a {contract_type} contract. Price includes all taxes, fees, labor, materials, equipment, overhead, and profit.
2.2 Schedule of Values: Before first payment application, Contractor shall submit a Schedule of Values allocating Contract Price to Work portions, subject to Owner's approval.
2.3 Applications for Payment: Monthly, by [DAY_FOR_PAYMENT_SUBMISSION] of each month, Contractor shall submit itemized Applications for Payment based on SOV, for Work completed and materials suitably stored. Applications must be accompanied by lien waivers and other documentation as Owner may reasonably require (e.g., certified payroll [IF_APPLICABLE]).
2.4 Review and Payment: Owner/Architect will review Applications within [DAYS_FOR_REVIEW] days. Approved amounts, less retainage, paid within [DAYS_FOR_PAYMENT_POST_APPROVAL] days of approval.
2.5 Retainage: Owner will retain {retainage_percentage} from each progress payment. Retainage released upon Final Completion, acceptance of Work, and satisfaction of closeout requirements.
2.6 Final Payment: Made after Contractor achieves Final Completion, submits all deliverables (as-builts, warranties, O&M manuals, final lien waivers), and Owner accepts the Work.""",
                ContractArticle.DEFAULT_TERMINATION.value: """ARTICLE - DEFAULT AND TERMINATION
3.1 Contractor Default: Events of Contractor default include (but not limited to): persistent failure to supply skilled workers/proper materials; failure to pay Subcontractors; disregard for laws/ordinances; failure to prosecute Work with diligence; insolvency.
3.2 Owner's Remedies: Upon Contractor default, Owner may, after [CURE_NOTICE_PERIOD_DAYS] days' written notice and opportunity to cure:
  a) Terminate Contractor's right to proceed with Work.
  b) Take possession of Site and materials; finish Work by reasonable means. Contractor liable for excess completion costs.
  c) Withhold payments due.
3.3 Termination by Owner for Convenience: Owner may terminate this Contract, in whole or part, for convenience upon [CONVENIENCE_TERMINATION_NOTICE_DAYS] days' written notice. Contractor shall be paid for Work properly executed to date of termination, reasonable demobilization costs, but not anticipated profit on uncompleted Work.
3.4 Liquidated Damages for Delay: If Contractor fails to achieve Substantial Completion by the agreed Contract Time (as adjusted by Change Orders), Contractor shall pay Owner {liquidated_damages_per_day} per calendar day of delay as liquidated damages, not as a penalty.""",
                ContractArticle.INSURANCE_BONDING.value: """ARTICLE - INSURANCE AND BONDING REQUIREMENTS
4.1 General Insurance: Contractor shall procure and maintain, at its sole expense, insurance coverages specified herein with insurers rated A-VII or better by A.M. Best, licensed in {governing_law}. Owner, its officers, and employees shall be named as additional insureds on CGL and Auto policies. Contractor shall provide Certificates of Insurance and endorsements to Owner before commencing Work. All policies shall provide for [DAYS_NOTICE_FOR_CANCELLATION] days' written notice to Owner of cancellation or material change.
4.2 Required Insurance Coverages:
{insurance_requirements_list}
4.3 Bonding Requirements: Contractor shall furnish the following surety bonds from a surety acceptable to Owner:
{bond_requirements_list}
4.4 Failure to Maintain: If Contractor fails to procure/maintain required insurance/bonds, Owner may procure same and deduct cost from payments due Contractor, or declare default.""",
                ContractArticle.SIGNATURE_BLOCKS.value: """ARTICLE - SIGNATURE BLOCKS
IN WITNESS WHEREOF, the parties hereto, intending to be legally bound, have executed this Agreement by their duly authorized representatives as of the Effective Date first written in this Agreement (or if no such date, the date of last signature below).

OWNER: {owner_name}                        CONTRACTOR: {contractor_name}

By: ____________________________________             By: ____________________________________
   (Signature)                                          (Signature)
Name: {owner_representative}                    Name: {contractor_representative}
Title: [OWNER_REPRESENTATIVE_TITLE]                  Title: [CONTRACTOR_REPRESENTATIVE_TITLE]
Date: _________________________________            Date: _________________________________
"""
            }
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    #Provides a resilient, general-purpose fallback template for a given article type, using broadly applicable contract placeholders.
    def _get_generic_template(self, article_type: str) -> str:
        try:
            return f"""ARTICLE - {article_type.upper()}
1.1 General Provisions for {article_type}
This Article outlines key terms and conditions related to {article_type}. All activities and obligations hereunder shall be performed in compliance with applicable laws, regulations, and the overall terms of this Agreement dated {{contract_date}} between {{owner_name}} and {{contractor_name}} for the {{project_name}}. The Project is of {{complexity}} complexity with an estimated duration of {{project_duration}}.
1.2 Specific Requirements Pertaining to {article_type}
Specific requirements concerning {article_type} include, but are not limited to, adherence to quality standards as detailed in {{quality_standards_list}}, compliance with all safety protocols per {{safety_requirements_list}}, and observance of building codes {{building_codes_list}}.
1.3 Further Details and Coordination
Further details regarding {article_type} may be found in specific appendices or addenda to this Agreement (e.g., [APPENDIX_REFERENCE_FOR_{article_type.replace(' ','_').upper()}] if applicable) or will be provided by the Owner's designated Representative, {{owner_representative}}. All parties agree to act in good faith and cooperate fully to fulfill the intent and requirements of this Article."""
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

class SyntheticContractGenerator:
    """Main orchestrator for synthetic contract generation."""
    #Initializes all core components required for contract generation, including synthetic data creation, validation, LLM content generation, and fallback handling, with optional LLM API key and seed for reproducibility.
    def __init__(self, api_key: Optional[str] = None, seed: Optional[int] = None):
        try:
            self.synthetic_generator = SyntheticDataGenerator(seed)
            self.content_validator = ContentValidator()
            self.llm_generator = LLMContentGenerator(api_key)
            self.fallback_generator = FallbackGenerator()
            self.generation_stats = {
                'total_contracts_generated_session': 0, 'llm_articles_attempted': 0,
                'llm_articles_successful': 0, 'fallback_articles_used': 0,
                'pii_validation_issues_found': 0, 'content_generation_failures': 0
            }
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise

    #Generates a complete contract by synthesizing base data, producing requested articles with LLM or fallback, validating content, assembling final text, and returning the contract with detailed stats and logs.
    def generate_complete_contract(self, requirements: ContractRequirements) -> Dict[str, Any]:
        try:
            logger.info(f"Starting contract generation for: {requirements.project_type}")
            current_run_stats = {'llm_articles_attempted': 0, 'llm_articles_successful': 0, 'fallback_articles_used': 0, 'pii_validation_issues_found': 0, 'content_generation_failures': 0}

            with st.status("Generating synthetic contract...", expanded=True) as status_bar:
                status_bar.write("1. Generating base synthetic contract data...")
                contract_data = self.synthetic_generator.generate_contract_data(requirements)
                
                generated_articles = {}
                all_contract_issues_log = []
                total_articles = len(requirements.selected_articles)

                for i, article_name in enumerate(requirements.selected_articles):
                    status_bar.write(f"2.{i+1}/{total_articles} Generating: '{article_name}'...")
                    article_content, pii_issues, generation_failed = self._generate_single_article(
                        article_name, requirements, contract_data, current_run_stats
                    )
                    generated_articles[article_name] = article_content
                    all_contract_issues_log.extend(pii_issues)
                    if pii_issues:
                        for issue_msg in pii_issues: status_bar.write(f"   ⚠️ Validation: {issue_msg}")
                    if generation_failed:
                        all_contract_issues_log.append(f"Generation FAILED for article: {article_name}")
                        status_bar.write(f" ❌ Generation FAILED for '{article_name}'. Check logs.")
                    else:
                        status_bar.write(f" ✅ Content for '{article_name}' generated.")
                
                status_bar.write("3. Assembling complete contract document...")
                complete_contract_text_raw = self._assemble_complete_contract(generated_articles, contract_data, requirements)
                
                status_bar.write("4. Final validation pass on full contract...")
                final_contract_text, final_pass_pii_issues = self.content_validator.validate_and_anonymize_content(complete_contract_text_raw)
                all_contract_issues_log.extend(final_pass_pii_issues)
                current_run_stats['pii_validation_issues_found'] += len(final_pass_pii_issues)
                if final_pass_pii_issues: status_bar.write(f" ⚠️ Final validation found {len(final_pass_pii_issues)} more PII/sensitive items.")
                
                self.generation_stats['total_contracts_generated_session'] += 1
                for key in current_run_stats:
                    self.generation_stats[key] = self.generation_stats.get(key,0) + current_run_stats[key]

                status_bar.write("✅ Contract generation complete!")
                status_bar.update(label="Contract Generation Successful!", state="complete")

                return {
                    'contract_text': final_contract_text, 'contract_data': asdict(contract_data),
                    'requirements_used': asdict(requirements), 'all_issues_log': all_contract_issues_log,
                    'metadata': {
                        'generation_timestamp_utc': datetime.now(UTC).isoformat() + "Z",
                        'articles_requested_count': total_articles,
                        'articles_generated_count': len(generated_articles),
                        'pii_validation_issues_count_this_run': current_run_stats['pii_validation_issues_found'],
                        'content_generation_failure_count_this_run': current_run_stats['content_generation_failures'],
                        'estimated_word_count': len(final_contract_text.split()),
                        'primary_generation_method': 'LLM (Fallback available)' if self.llm_generator.is_initialized else 'Template-based Only'
                    },
                    'generation_run_stats': current_run_stats 
                }
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            if 'st' in sys.modules and 'status_bar' in locals(): status_bar.update(label="Contract Generation Failed!", state="error")
            raise

    # Generates and validates content for a single article using LLM or fallback templates, tracking stats and errors.
    def _generate_single_article(self, article_name: str, requirements: ContractRequirements,
                               contract_data: GeneratedContractData, run_stats: Dict[str, int]) -> Tuple[str, List[str], bool]:
        try:
            pii_validation_issues = []
            raw_content = ""
            generation_failed_flag = False

            if self.llm_generator.is_initialized and self.llm_generator.model:
                run_stats['llm_articles_attempted'] += 1
                try:
                    raw_content = self.llm_generator.generate_article_content(article_name, requirements, contract_data)
                    if raw_content and not raw_content.strip().startswith("ARTICLE -") and \
                       "[Content generation failed" not in raw_content and \
                       "[CRITICAL ERROR: Fallback generation mechanism failed" not in raw_content and \
                       "[Fallback content generation for this article encountered an error" not in raw_content:
                        run_stats['llm_articles_successful'] += 1
                    else:
                        logger.warning(f"LLM for '{article_name}' likely used internal fallback or failed to produce distinct content. Forcing explicit fallback.")
                        raw_content = "" 
                        run_stats['fallback_articles_used'] += 1
                except Exception as e_llm:
                    logger.error(f"LLM generation for '{article_name}' exception: {e_llm}. Using fallback.", exc_info=True)
                    run_stats['fallback_articles_used'] += 1
                    raw_content = ""
            else:
                run_stats['fallback_articles_used'] += 1
                raw_content = ""
            if not raw_content.strip():
                try:
                    logger.info(f"Executing explicit fallback generation for article: {article_name}")
                    raw_content = self.fallback_generator.generate_fallback_article(article_name, requirements, contract_data)
                    if "[Content generation for this article encountered an error" in raw_content or \
                       "[Fallback content generation for this article encountered an error" in raw_content or \
                       "[CRITICAL ERROR: Fallback generation mechanism failed" in raw_content or \
                       "[Fallback content generation failed due to template key error" in raw_content:
                        logger.error(f"Fallback generation for {article_name} reported an internal error in its output.")
                        generation_failed_flag = True
                        run_stats['content_generation_failures'] += 1
                except Exception as fallback_e:
                    logger.error(f"Explicit fallback generation critically failed for '{article_name}': {fallback_e}", exc_info=True)
                    raw_content = f"ARTICLE - {article_name.upper()}\n\n[CRITICAL FAILURE: Both LLM (if attempted) and explicit fallback generation failed for this article. Error: {str(fallback_e)}]"
                    generation_failed_flag = True
                    run_stats['content_generation_failures'] += 1
            
            if not raw_content.strip():
                logger.error(f"FATAL: No content at all generated for article {article_name}.")
                raw_content = f"ARTICLE - {article_name.upper()}\n\n[ULTIMATE ERROR: NO CONTENT WAS GENERATED. MANUAL INPUT REQUIRED.]"
                generation_failed_flag = True
                run_stats['content_generation_failures'] += 1

            validated_content, pii_issues_from_validation = self.content_validator.validate_and_anonymize_content(raw_content)
            pii_validation_issues.extend(pii_issues_from_validation)
            return validated_content, pii_validation_issues, generation_failed_flag
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)} for article {article_name}: {str(e)}", exc_info=True)
            run_stats['content_generation_failures'] +=1
            return f"ARTICLE - {article_name.upper()}\n\n[UNHANDLED EXCEPTION during generation of this article: {str(e)}]", [], True

    # Assembles the full contract text by combining a formatted header, table of contents, generated articles, and a closing footer.
    def _assemble_complete_contract(self, articles: Dict[str, str],
                                  contract_data: GeneratedContractData,
                                  requirements: ContractRequirements) -> str:
        try:
            header = f"""================================================================================
SYNTHETIC CONSTRUCTION AGREEMENT
================================================================================
Contract Identification Number: [CONTRACT_ID_PLACEHOLDER_{random.randint(10000,99999)}]
Date of Agreement: {contract_data.contract_date}

Between:
The Owner:
    Name:           {contract_data.owner_name}
    Represented by: {contract_data.owner_representative}
    Address:        [OWNER_ADDRESS_LINE_1_PLACEHOLDER], [OWNER_CITY_PLACEHOLDER], [OWNER_STATE_PROVINCE_PLACEHOLDER] [OWNER_POSTAL_CODE_PLACEHOLDER]
And
The Contractor:
    Name:           {contract_data.contractor_name}
    Represented by: {contract_data.contractor_representative}
    Address:        {contract_data.contractor_address}

For The Project:
    Project Name:        {contract_data.project_name}
    Project Location:    {requirements.location}
    Project Description: {contract_data.project_description}

Contract Sum: {contract_data.contract_amount} ({contract_data.contract_type})
Contract Duration: {contract_data.project_duration}
    Anticipated Start Date:     {contract_data.start_date}
    Anticipated Completion Date: {contract_data.completion_date}

Governing Law: {contract_data.governing_law}
--------------------------------------------------------------------------------
TABLE OF CONTENTS (Illustrative - Actual articles follow in order of selection)
--------------------------------------------------------------------------------
"""
            contract_parts = [header]
            toc_items = [f"Article {i+1} : {name}" for i, name in enumerate(requirements.selected_articles)]
            contract_parts.append("\n".join(toc_items))
            contract_parts.append("--------------------------------------------------------------------------------\n")

            for i, article_name_key in enumerate(requirements.selected_articles):
                content = articles.get(article_name_key, f"\nARTICLE {i+1} : {article_name_key.upper()}\n\n[CRITICAL ERROR: CONTENT FOR THIS ARTICLE WAS NOT FOUND IN GENERATED ARTICLES DICTIONARY - PLACEHOLDER]\n")
                contract_parts.append(f"\n\n================================================================================\nARTICLE {i+1} : {article_name_key.upper()}\n================================================================================\n{content.strip()}\n")
            
            footer = f"""================================================================================
END OF ARTICLES
================================================================================
IN WITNESS WHEREOF, the parties hereto have executed this Agreement by their duly authorized representatives as of the date first written above.

THE OWNER:                                       THE CONTRACTOR:
{contract_data.owner_name}                              {contract_data.contractor_name}
"""
            contract_parts.append(footer)
            return "\n".join(contract_parts)
        except Exception as e:
            logger.error(f"Error in {get_error_location(self)}: {str(e)}", exc_info=True)
            raise
import requests
from urllib.parse import quote
api_key='f8b80c730dedf2d2a1e44288068ca339'
def get_coordinates(location):
    """Get latitude and longitude for a given location"""
    try:
        if not location or location.strip() == '':
            return None, None, "Please enter a location"
        encoded_location = quote(location.strip())
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={encoded_location}&limit=1&appid={api_key}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or len(data) == 0:
            return None, None, f"Location '{location}' not found"
        
        location_data = data[0]
        latitude = location_data.get('lat')
        longitude = location_data.get('lon')
        
        return latitude, longitude, None
        
    except requests.exceptions.Timeout:
        return None, None, "Request timeout - please try again"
    except requests.exceptions.HTTPError:
        return None, None, "API error - please check your API key"
    except Exception as e:
        return None, None, f"Error: {str(e)}"


#Transforms contract text by applying regex-based markdown formatting for placeholders, article headers, separators, lists, and lettered sub-items
def preprocess_markdown(text: str) -> str:
    try:
        text = re.sub(r'(\[[A-Z0-9_]+(?:_[A-Z0-9_]+)*\])', r'**\1**', text)
        text = re.sub(r'^ARTICLE\s+(\d+)\s+:\s+(.*)', r'### Article \1: \2', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^(?:={5,}|-{5,})\s*$', r'---', text, flags=re.MULTILINE)
        text = re.sub(r'^\s{2,}-\s+(.*)', r'* \1', text, flags=re.MULTILINE)
        text = re.sub(r'^\s{2,}([a-z])\)\s+(.*)', lambda m: f"  {ord(m.group(1).lower()) - ord('a') + 1}. {m.group(2)}", text, flags=re.MULTILINE)
        return text
    except Exception as e:
        logger.error(f"Error in {get_error_location()}: {str(e)}", exc_info=True) # No class instance
        raise

#Builds a multi-tabbed Streamlit UI for configuring, generating, viewing, analyzing, and exporting synthetic construction contracts using LLM or templates.
def create_streamlit_interface():
    try:
        st.set_page_config(page_title="Synthetic Contract Generator", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")
        st.title(" Enhanced Synthetic Construction Contract Generator")
        st.caption(f"Version 1.0.0 - Author: Mervin Joseph L - Date: {datetime.now().strftime('%B %Y')}")
        st.markdown("---")
        for key, default_val in [('generated_contract_result', None), ('generation_log', [])]:
            if key not in st.session_state: st.session_state[key] = default_val

        with st.sidebar:
            st.header("⚙️ Configuration")
            api_key_input = st.text_input("Google Gemini API Key (Optional)", type="password", help="Enables LLM-powered content. If empty, uses templates.")
            if api_key_input: st.session_state.api_key = api_key_input # Store/update
            elif 'api_key' in st.session_state and not api_key_input: del st.session_state.api_key # Clear if emptied
            
            use_seed_checkbox = st.checkbox("Set Random Seed for Reproducibility", value=False) # Default False
            seed_value_input = None
            if use_seed_checkbox:
                seed_value_input = st.number_input("Random Seed Value", value=42, min_value=1, max_value=100000, step=1, key="seed_value_input_key")
            st.markdown("---"); st.subheader("About")
            st.info("This tool generates synthetic, anonymized construction contract text using Google Gemini LLM (if API key provided) or internal templates.")

        tab_setup, tab_generated_contract, tab_analytics, tab_markdown = st.tabs(["📝 Contract Setup", "📄 Generated Contract", "📊 Analytics", "👁️ Markdown Preview"])

        with tab_setup:
            st.header("Contract Requirements Definition")
            with st.form("contract_requirements_form"):
                col1, col2 = st.columns(2)
                with col1:
                    project_type_input = st.selectbox("Project Type", [ptype.value for ptype in ProjectComplexity] if False else ["Commercial Office Building", "Residential Development Complex", "Highway Construction", "Bridge Design", "Water Treatment Facility", "Educational Building", "Public Safety Complex", "Airport Development", "Transit Expansion", "Environmental Remediation", "Energy Installation", "Industrial Facility"], index=0, key="project_type_key")
                    contract_value_input = st.number_input("Contract Value (USD)", 50000, 100_000_000, 2_500_000, 100_000, "%d", key="contract_value_key")
                    complexity_options_map = {pc.name.replace("_", "-"): pc.value for pc in ProjectComplexity}
                    complexity_display = st.select_slider("Project Complexity", list(complexity_options_map.keys()), "MODERATE", key="complexity_slider_key")
                    complexity_input = complexity_options_map[complexity_display]
                with col2:
                    duration_days_input = st.number_input("Project Duration (Days)", 30, 1825, 365, 15, key="duration_days_key")

                    # --- Location input ---
                    location_input = st.text_input("Project General Location", "Southern District, Province Gamma", key="location_key")

                    # --- Call API when user presses Enter (i.e., when input changes) ---
                    if location_input:
                        lat, lon, error = get_coordinates(location_input)
                        if error:
                            st.warning(error)
                        else:
                            st.success(f"Coordinates: Latitude = {lat}, Longitude = {lon}")
                            st.session_state["latitude"] = lat
                            st.session_state["longitude"] = lon

                    # --- Special requirements ---
                    special_requirements_input = st.text_area(
                        "Special Requirements (one per line, optional)",
                        placeholder="e.g., LEED Gold Certification\nPhased Occupancy Required",
                        height=100,
                        key="special_req_key"
                    )
                st.subheader("Select Contract Articles to Generate")
                all_article_options_display = [article.value for article in ContractArticle]
                default_selected_articles_display = [ContractArticle.DEFINITIONS.value, ContractArticle.SCOPE_OF_WORK.value, ContractArticle.CONTRACT_PRICE.value, ContractArticle.TIME_PERFORMANCE.value, ContractArticle.INSURANCE_BONDING.value, ContractArticle.DEFAULT_TERMINATION.value, ContractArticle.SIGNATURE_BLOCKS.value]
                selected_articles_display_names = st.multiselect("Choose articles:", all_article_options_display, default=default_selected_articles_display, key="multiselect_articles_key")
                submitted = st.form_submit_button("✨ Generate Synthetic Contract ✨", use_container_width=True, type="primary")

            if submitted:
                if not selected_articles_display_names: st.error("Please select at least one contract article.")
                else:
                    st.session_state.generation_log = ["Generation process initiated..."]
                    with st.spinner("Hold tight! Crafting your synthetic contract... This may take a moment, especially with LLM."):
                        try:
                            current_requirements = ContractRequirements(project_type_input, int(contract_value_input), int(duration_days_input), location_input, complexity_input, selected_articles_display_names, special_requirements_input)
                            current_api_key = st.session_state.get('api_key', None) # Get from session state
                            contract_gen_instance = SyntheticContractGenerator(current_api_key, seed_value_input if use_seed_checkbox else None)
                            if current_api_key and not contract_gen_instance.llm_generator.is_initialized: st.warning("LLM initialization failed (check API key/console). Proceeding with template-based generation.", icon="⚠️")
                            elif not current_api_key: st.info("No API key provided. Using template-based generation for all articles.", icon="📄")
                            result_dict = contract_gen_instance.generate_complete_contract(current_requirements)
                            st.session_state.generated_contract_result = result_dict
                            st.session_state.generation_log.append("Contract generation completed successfully!")
                            st.success(f"Synthetic contract generated! PII/Validation issues found: {result_dict['metadata']['pii_validation_issues_count_this_run']}. Content Generation Failures: {result_dict['metadata']['content_generation_failure_count_this_run']}.")
                        except Exception as e_form_submit:
                            st.session_state.generation_log.append(f"Error: {str(e_form_submit)}")
                            logger.error("Error during Streamlit form submission block:", exc_info=True)
                            st.error(f"💥 An unexpected error occurred during contract generation: {str(e_form_submit)}")
                            st.exception(e_form_submit) # Shows full traceback for dev
        
        with tab_generated_contract:
            st.header("View Generated Contract")
            if st.session_state.generated_contract_result:
                res = st.session_state.generated_contract_result; meta = res['metadata']
                st.subheader("Contract Overview & Generation Stats")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Articles Generated", meta['articles_generated_count'], help="Number of articles for which content was generated.")
                col2.metric("Est. Word Count", meta['estimated_word_count'], help="Approximate word count of the full contract text.")
                col3.metric("PII/Validation Issues", meta['pii_validation_issues_count_this_run'], help="Potential PII or sensitive data patterns flagged and anonymized in this run.")
                col4.metric("Content Failures", meta['content_generation_failure_count_this_run'], help="Number of articles where content generation failed (LLM and Fallback).")

                if res['all_issues_log']:
                    with st.expander("Show Full Generation & Validation Log (includes PII anonymization messages)", expanded=False):
                        for issue_entry in res['all_issues_log']: st.warning(f"LOG: {issue_entry}")
                
                st.subheader("Full Contract Text (Editable)")
                editable_contract_text = st.text_area("Edit Contract Text:", value=res['contract_text'], height=600, key="editable_contract_text_area_key", help="You can make changes to the text here before downloading.")
                if editable_contract_text != res['contract_text']: # If user edits
                     st.session_state.generated_contract_result['contract_text'] = editable_contract_text # Update state
                
                current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(label="📥 Download as TXT", data=editable_contract_text, file_name=f"synthetic_contract_{current_time_str}.txt", mime="text/plain", use_container_width=True, key="download_txt_key")
            else: st.info("No contract has been generated yet. Please use the 'Contract Setup' tab first.")

        with tab_analytics:
            st.header("Generation Analytics & Details")
            if st.session_state.generated_contract_result:
                res = st.session_state.generated_contract_result
                stats = res.get('generation_run_stats', {})

                st.subheader("Statistics for Last Generation Run")
                if stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("LLM Articles Attempted", stats.get('llm_articles_attempted', 'N/A'))
                    with col2:
                        st.metric("LLM Articles Successful", stats.get('llm_articles_successful', 'N/A'))
                    with col3:
                        st.metric("Fallback Articles Used", stats.get('fallback_articles_used', 'N/A'))
                else:
                    st.write("Detailed run statistics not available for this generation.")

                st.subheader("Core Contract Data (Synthetic - Generated)")
                contract_data_df = pd.DataFrame(list(res['contract_data'].items()), columns=['Field', 'Value'])
                if 'Value' in contract_data_df.columns and contract_data_df['Value'].dtype == 'object':
                     contract_data_df['Value'] = contract_data_df['Value'].astype(str)
                st.dataframe(contract_data_df, use_container_width=True, hide_index=True, key="df_contract_data_key")

            else:
                st.info("Generate a contract first to see analytics and details.")

        with tab_markdown:
            st.header("Markdown Preview of Contract")
            if st.session_state.generated_contract_result:
                contract_text_for_md = st.session_state.generated_contract_result.get('contract_text', "")
                if contract_text_for_md:
                    processed_markdown = preprocess_markdown(contract_text_for_md)
                    st.markdown(processed_markdown, unsafe_allow_html=False)
                    st.download_button(label="📥 Download as Markdown (.md)", data=processed_markdown, file_name=f"synthetic_contract_markdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown", use_container_width=True, key="download_md_key")
                else: st.warning("Contract text is currently empty. Cannot generate Markdown preview.")
            else: st.info("No contract generated yet. Use the 'Contract Setup' tab to create a contract.")
    except Exception as e_interface:
        logger.critical(f"Fatal error in create_streamlit_interface: {e_interface}", exc_info=True)
        st.error(f"A critical application error occurred in the UI: {e_interface}. Please check the application logs for more details.")


if __name__ == "__main__":
    try:
        create_streamlit_interface()
    except Exception as e_main_block:
        logger.critical(f"Fatal error running Streamlit application in __main__ block: {e_main_block}", exc_info=True)
        if 'st' in sys.modules and hasattr(st, 'error'):
            st.error(f"A critical application error occurred preventing the application from running: {e_main_block}. Please consult the logs.")
        else:
            print(f"FATAL ERROR in __main__ block: {e_main_block}")
