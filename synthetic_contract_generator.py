"""
Enhanced Synthetic Construction Contract Data Generator
=====================================================

A comprehensive system for generating synthetic construction contracts with:
- Real-time data validation and anonymization
- Interactive Streamlit interface
- User-defined contract sections
- LLM-powered content generation with fallback mechanisms
- Privacy-preserving synthetic data generation

Author: AI Assistant
Date: June 2025
"""

import os
import re
import json
import logging
import random
import streamlit as st
import sys
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import google.generativeai as genai
from faker import Faker
import pandas as pd

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """
    Data structure containing all requirements for contract generation.
    Includes project details, compliance requirements, and risk factors.
    """
    project_type: str
    contract_value: int
    duration_days: int
    location: str
    complexity: str
    special_requirements: List[str] = field(default_factory=list)
    regulatory_requirements: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    selected_articles: List[str] = field(default_factory=list)

@dataclass
class GeneratedContractData:
    """
    Comprehensive data structure for all generated contract information.
    Contains anonymized parties, financial terms, and legal requirements.
    """
    # Basic contract information
    contract_date: str
    project_name: str
    project_description: str
    
    # Contract parties with anonymized details
    owner_name: str
    owner_representative: str
    contractor_name: str
    contractor_address: str
    contractor_representative: str
    
    # Financial terms and payment structure
    contract_amount: str
    contract_type: str
    payment_terms: str
    retainage_percentage: str
    
    # Project timeline and milestones
    start_date: str
    completion_date: str
    project_duration: str
    
    # Legal and compliance requirements
    governing_law: str
    dispute_resolution: str
    insurance_requirements: List[str] = field(default_factory=list)
    bond_requirements: List[str] = field(default_factory=list)
    
    # Technical specifications and standards
    building_codes: List[str] = field(default_factory=list)
    quality_standards: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)

class SyntheticDataGenerator:
    """
    Generates realistic but completely synthetic data for construction contracts.
    Ensures no real-world entities are used while maintaining data realism.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic data generator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for consistent data generation across runs
        """
        if seed is None:
            seed = random.randint(1, 10000)
        
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        
        # Define realistic but generic project types
        self.project_types = [
            "Highway Construction and Reconstruction",
            "Bridge Design and Construction", 
            "Municipal Water Treatment Facility",
            "Educational Institution Building",
            "Public Safety Complex",
            "Airport Infrastructure Development",
            "Transit System Expansion",
            "Environmental Remediation Project",
            "Energy Infrastructure Installation",
            "Telecommunications Network Deployment",
            "Residential Development Complex",
            "Commercial Office Building",
            "Industrial Manufacturing Facility"
        ]
        
        # Generic location templates
        self.location_templates = [
            "Metropolitan Area Alpha", "City Beta Region", "County Gamma District",
            "Township Delta Zone", "Municipal Area Epsilon", "Regional Sector Zeta"
        ]
        
    def generate_contract_data(self, requirements: ContractRequirements) -> GeneratedContractData:
        """
        Generate comprehensive synthetic contract data based on provided requirements.
        All generated data is anonymized and contains no real-world identifiers.
        
        Args:
            requirements: Contract requirements specifying project parameters
            
        Returns:
            GeneratedContractData: Complete synthetic contract information
        """
        
        # Generate realistic but anonymous dates
        contract_date = self.fake.date_between(start_date='-3m', end_date='+1m')
        start_date = contract_date + timedelta(days=random.randint(15, 60))
        completion_date = start_date + timedelta(days=requirements.duration_days)
        
        # Create anonymous party names using generic templates
        contractor_types = ["Construction", "Builders", "Engineering", "Development", "Infrastructure"]
        contractor_entities = ["LLC", "Inc.", "Corp.", "Co."]
        
        # Generate anonymous but realistic project name
        project_suffix = self.fake.catch_phrase().replace(self.fake.company(), "Generic Corp")
        
        return GeneratedContractData(
            # Basic contract information with anonymized details
            contract_date=contract_date.strftime("%B %d, %Y"),
            project_name=f"{requirements.project_type} - Phase {random.randint(1,5)}",
            project_description=f"Comprehensive {requirements.project_type.lower()} project including all associated work and deliverables",
            
            # Anonymized party information
            owner_name=f"Municipal Authority Alpha",
            owner_representative=f"[OWNER_REP_NAME], P.E.",
            contractor_name=f"[CONTRACTOR_COMPANY] {random.choice(contractor_entities)}",
            contractor_address=f"[CONTRACTOR_ADDRESS], [CITY], [STATE] [ZIP]",
            contractor_representative=f"[CONTRACTOR_REP_NAME]",
            
            # Financial terms
            contract_amount=f"${requirements.contract_value:,}.00",
            contract_type=random.choice(list(ContractType)).value,
            payment_terms="Net 30 days upon approved invoice",
            retainage_percentage=f"{random.choice([5, 10, 15])}%",
            
            # Timeline information
            start_date=start_date.strftime("%B %d, %Y"),
            completion_date=completion_date.strftime("%B %d, %Y"),
            project_duration=f"{requirements.duration_days} calendar days",
            
            # Legal framework
            governing_law="State of [JURISDICTION]",
            dispute_resolution="Binding arbitration in accordance with [STATE] law",
            
            # Insurance requirements scaled by project value
            insurance_requirements=self._generate_insurance_requirements(requirements.contract_value),
            bond_requirements=self._generate_bond_requirements(requirements.contract_value),
            
            # Technical standards
            building_codes=self._generate_building_codes(),
            quality_standards=self._generate_quality_standards(),
            safety_requirements=self._generate_safety_requirements()
        )
    
    def _generate_insurance_requirements(self, contract_value: int) -> List[str]:
        """Generate appropriate insurance requirements based on contract value."""
        base_requirements = [
            "General Liability Insurance - $2,000,000 per occurrence",
            "Professional Liability Insurance - $1,000,000 per claim", 
            "Workers' Compensation as required by law",
            "Commercial Auto Liability - $1,000,000 combined single limit"
        ]
        
        if contract_value > 5000000:
            base_requirements.extend([
                "Umbrella Liability - $5,000,000",
                "Environmental Liability - $1,000,000"
            ])
        
        return base_requirements
    
    def _generate_bond_requirements(self, contract_value: int) -> List[str]:
        """Generate bond requirements based on contract value thresholds."""
        if contract_value < 100000:
            return []
        
        requirements = [
            "Performance Bond - 100% of contract value",
            "Payment Bond - 100% of contract value"
        ]
        
        if contract_value > 1000000:
            requirements.append("Maintenance Bond - 2 years from completion")
        
        return requirements
    
    def _generate_building_codes(self) -> List[str]:
        """Generate standard building codes applicable to construction projects."""
        return [
            "International Building Code (IBC) Current Edition",
            "International Fire Code (IFC) Current Edition",
            "Americans with Disabilities Act (ADA) Standards",
            "Local Municipal Building Codes and Ordinances"
        ]
    
    def _generate_quality_standards(self) -> List[str]:
        """Generate industry-standard quality requirements."""
        return [
            "ASTM International Standards",
            "American Institute of Steel Construction (AISC)",
            "American Concrete Institute (ACI) Standards",
            "National Institute of Standards and Technology (NIST)"
        ]
    
    def _generate_safety_requirements(self) -> List[str]:
        """Generate comprehensive safety requirements."""
        return [
            "Site-specific Safety Plan required prior to work commencement",
            "Daily safety meetings mandatory for all personnel",
            "Personal Protective Equipment (PPE) compliance verification",
            "Environmental protection and waste management plan"
        ]

class ContentValidator:
    """
    Validates generated content for real-world data leakage and ensures anonymization.
    Implements multiple validation layers for comprehensive privacy protection.
    """
    
    def __init__(self):
        """Initialize the content validator with comprehensive detection patterns."""
        
        # Patterns for detecting potentially real data
        self.sensitive_patterns = {
            'phone_numbers': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn_numbers': r'\b\d{3}-\d{2}-\d{4}\b',
            'real_company_names': self._get_real_company_patterns(),
            'real_person_names': self._get_common_name_patterns(),
            'specific_addresses': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)\b',
            'license_numbers': r'\b[A-Z]{2}\d{6,8}\b',
            'bank_routing': r'\b\d{9}\b'
        }
        
        # Replacement templates for anonymization
        self.replacement_templates = {
            'phone_numbers': '[PHONE_NUMBER]',
            'email_addresses': '[EMAIL_ADDRESS]', 
            'ssn_numbers': '[SSN]',
            'real_company_names': '[COMPANY_NAME]',
            'real_person_names': '[PERSON_NAME]',
            'specific_addresses': '[ADDRESS]',
            'license_numbers': '[LICENSE_NUMBER]',
            'bank_routing': '[ROUTING_NUMBER]'
        }
    
    def validate_and_anonymize_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Comprehensive validation and anonymization of generated content.
        
        Args:
            content: Raw generated content to validate and clean
            
        Returns:
            Tuple of (anonymized_content, list_of_issues_found)
        """
        issues_found = []
        cleaned_content = content
        
        # Ensure content is a string
        if not isinstance(cleaned_content, str):
            cleaned_content = str(cleaned_content)
            issues_found.append("Content was not a string, converted to string")

        # Apply each validation pattern
        for pattern_name, pattern_regex in self.sensitive_patterns.items():
            matches = re.finditer(pattern_regex, cleaned_content, re.IGNORECASE)
            match_count = 0
            
            for match in matches:
                match_count += 1
                matched_text = match.group()
                replacement = self.replacement_templates[pattern_name]
                cleaned_content = cleaned_content.replace(matched_text, replacement)
                
            if match_count > 0:
                issues_found.append(f"Found and anonymized {match_count} instances of {pattern_name}")
        
        # Additional context-aware validation
        additional_issues = self._validate_construction_context(cleaned_content)
        issues_found.extend(additional_issues)
        
        return cleaned_content, issues_found
    
    def _get_real_company_patterns(self) -> str:
        """Generate pattern for detecting real company names."""
        # Common real company indicators
        real_company_indicators = [
            r'\b(?:Microsoft|Google|Apple|Amazon|Meta|Tesla|Ford|Boeing|Caterpillar|General\s+Electric)\b'
        ]
        return '|'.join(real_company_indicators)
    
    def _get_common_name_patterns(self) -> str:
        """Generate pattern for detecting common real person names."""
        # This is a simplified version - in production, use a comprehensive names database
        common_names = [
            r'\b(?:John|Jane|Michael|Sarah|David|Lisa|Robert|Jennifer|William|Jessica)\s+(?:Smith|Johnson|Williams|Brown|Jones|Garcia|Miller|Davis|Rodriguez|Martinez)\b'
        ]
        return '|'.join(common_names)
    
    def _validate_construction_context(self, content: str) -> List[str]:
        """Perform construction-specific validation checks."""
        issues = []
        
        # Check for overly specific technical details that might be proprietary
        if re.search(r'Patent\s+(?:No\.?\s*)?\d{7,}', content, re.IGNORECASE):
            issues.append("Found potential patent number reference")
        
        # Check for specific brand names that should be genericized
        brand_patterns = r'\b(?:Caterpillar|CAT|John\s+Deere|Volvo|Liebherr|Komatsu|Hitachi)\b'
        if re.search(brand_patterns, content, re.IGNORECASE):
            issues.append("Found specific equipment brand names - should be genericized")
        
        return issues

class LLMContentGenerator:
    """
    Manages LLM interactions for generating construction contract content.
    Handles prompts, responses, and error recovery mechanisms.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM content generator with API configuration.
        
        Args:
            api_key: Google Gemini API key for content generation
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = None
        self.is_initialized = False
        self._setup_model()
        
    def _setup_model(self) -> bool:
        """
        Configure and initialize the LLM model.
        
        Returns:
            bool: True if model setup successful, False otherwise
        """
        try:
            if not self.api_key:
                logger.warning("No API key provided. LLM generation will not be available.")
                return False
                
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.is_initialized = True
            logger.info("LLM model initialized successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            self.model = None
            self.is_initialized = False
            return False
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text (rough approximation)."""
        return len(text.split()) + len(text) // 4  # Rough estimate: words + 1 token per 4 characters
    
    def generate_article_content(self, article_type: str, requirements: ContractRequirements, 
                               contract_data: GeneratedContractData, max_retries: int = 3, 
                               max_tokens_per_call: int = 2500) -> str:
        """
        Generate contract article content using the LLM.
        Falls back to template-based generation if LLM is unavailable or fails.
        """
        if not self.model or not self.is_initialized:
            logger.warning("LLM model not available, using fallback generation")
            return self._fallback_generation(article_type, requirements, contract_data)
        
        initial_prompt = self._create_article_prompt(article_type, requirements, contract_data)
        generated_text = ""
        stop_reason = "length"
        iteration_count = 0
        max_iterations = 10
        
        # Use st.write instead of st.status to avoid nesting expanders
        st.write(f"Generating {article_type}...")
        
        while stop_reason == "length" and iteration_count < max_iterations:
            st.write(f"Generation iteration {iteration_count + 1} for {article_type}")
            iteration_count += 1
            
            if not generated_text:
                prompt = initial_prompt
            else:
                context_length = min(500, len(generated_text))
                context = generated_text[-context_length:]
                prompt = f"Continue from where this left off:\n\n{context}\n\nContinue writing seamlessly:"
            
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens_per_call,
                            temperature=0.7,
                            top_p=0.9
                        )
                    )
                    
                    new_content = response.text.strip()
                    if generated_text and new_content:
                        generated_text += "\n" + new_content
                    elif new_content:
                        generated_text = new_content
                    
                    generated_tokens_approx = self.estimate_tokens(new_content)
                    st.write(f"Generated {generated_tokens_approx} tokens in iteration {iteration_count}")
                    
                    token_threshold = max_tokens_per_call * 0.85
                    natural_endings = ['.', '!', '?', ':', ';', '"', "'", ')', ']', '}']
                    ends_naturally = any(new_content.rstrip().endswith(ending) for ending in natural_endings)
                    ends_mid_word = (len(new_content) > 0 and 
                                   not new_content.endswith(' ') and 
                                   not ends_naturally and
                                   new_content[-1].isalnum())
                    
                    if generated_tokens_approx >= token_threshold or ends_mid_word:
                        st.write("Response likely truncated - continuing...")
                        stop_reason = "length"
                    else:
                        st.write("Response appears complete - stopping...")
                        stop_reason = "stop"
                        break
                    
                except Exception as e:
                    logger.error(f"LLM generation failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        st.write(f"Failed after {max_retries} attempts. Falling back to template.")
                        return self._fallback_generation(article_type, requirements, contract_data)
            
            if stop_reason != "length":
                break
        
        if iteration_count >= max_iterations:
            st.write("Maximum iterations reached. Falling back to template.")
            return self._fallback_generation(article_type, requirements, contract_data)
        
        return generated_text
    
    def _fallback_generation(self, article_type: str, requirements: ContractRequirements,
                           contract_data: GeneratedContractData) -> str:
        """Generate article content using fallback templates."""
        fallback_generator = FallbackGenerator()
        return fallback_generator.generate_fallback_article(article_type, requirements, contract_data)
    
    def _create_article_prompt(self, article_type: str, requirements: ContractRequirements,
                             contract_data: GeneratedContractData) -> str:
        """
        Create specialized prompts for different contract articles.
        
        Args:
            article_type: Type of article to generate
            requirements: Contract requirements
            contract_data: Contract data for context
            
        Returns:
            str: Formatted prompt for LLM generation
        """
        base_context = f"""
You are a senior construction attorney specializing in public works contracts. 
Generate a comprehensive, legally sound contract article with the following specifications:

PROJECT CONTEXT:
- Project Type: {requirements.project_type}
- Contract Value: ${requirements.contract_value:,}
- Duration: {requirements.duration_days} days
- Complexity: {requirements.complexity}

IMPORTANT: Use only generic placeholders like [OWNER_NAME], [CONTRACTOR_NAME], 
[PROJECT_NAME], etc. Do not include any real company names, person names, or 
specific identifying information.
"""

        article_prompts = {
            "Scope of Work and Specifications": f"""
{base_context}

Generate a detailed SCOPE OF WORK AND SPECIFICATIONS article that includes:

1. PROJECT OVERVIEW
   - Clear description of work to be performed
   - Project objectives and deliverables
   - Work site conditions and constraints

2. TECHNICAL SPECIFICATIONS
   - Detailed work descriptions by phase
   - Material specifications and standards
   - Quality requirements and testing protocols
   - Performance criteria and acceptance standards

3. CONTRACTOR RESPONSIBILITIES
   - Supervision and project management
   - Equipment and material procurement
   - Subcontractor coordination requirements
   - Progress reporting and documentation

4. SPECIAL REQUIREMENTS
   - Environmental compliance measures
   - Safety protocols and procedures
   - Permit and approval responsibilities
   - Coordination with utilities and agencies

Make the scope specific and measurable for this {requirements.complexity} complexity project.
Use professional legal language with numbered subsections.
""",

            "Contract Price and Payment Terms": f"""
{base_context}

Generate a comprehensive CONTRACT PRICE AND PAYMENT TERMS article including:

1. CONTRACT PRICE
   - Total contract amount: {contract_data.contract_amount}
   - Price basis and inclusions/exclusions
   - Unit price breakdowns if applicable

2. PAYMENT SCHEDULE
   - Progress payment procedures
   - Invoice submission requirements
   - Approval and payment timelines
   - Retainage provisions ({contract_data.retainage_percentage})

3. CHANGE ORDER PROVISIONS
   - Change order procedures and approvals
   - Pricing methodology for additional work
   - Time impact considerations

4. FINAL PAYMENT
   - Substantial completion requirements
   - Final payment conditions
   - Warranty and guarantee provisions

Use precise legal language with clear procedures and deadlines.
""",

            "Default and Termination": f"""
{base_context}

Generate a comprehensive DEFAULT AND TERMINATION article covering:

1. EVENTS OF DEFAULT
   - Contractor default conditions
   - Owner default conditions
   - Material breach definitions

2. NOTICE AND CURE PROVISIONS
   - Notice requirements and methods
   - Cure periods and procedures
   - Continuing default consequences

3. REMEDIES UPON DEFAULT
   - Rights and remedies for each party
   - Liquidated damages provisions
   - Cover and completion rights

4. TERMINATION PROCEDURES
   - Termination for cause
   - Termination for convenience
   - Post-termination obligations

Include appropriate liquidated damages (suggest ${max(500, requirements.contract_value // 10000)} per day).
""",

            "Insurance and Bonding Requirements": f"""
{base_context}

Generate comprehensive INSURANCE AND BONDING REQUIREMENTS including:

1. REQUIRED INSURANCE COVERAGE
   - Types and minimum coverage amounts
   - Certificate of insurance requirements
   - Additional insured provisions
   - Insurance carrier requirements

2. BONDING REQUIREMENTS
   - Performance bond requirements
   - Payment bond requirements  
   - Maintenance bond provisions
   - Bond form and conditions

3. CLAIMS AND PROCEDURES
   - Claims notification procedures
   - Coordination between insurers
   - Waiver of subrogation provisions

4. COMPLIANCE MONITORING
   - Ongoing compliance verification
   - Remedies for non-compliance
   - Insurance updates and renewals

Base requirements on contract value of ${requirements.contract_value:,}.
"""
        }
        
        return article_prompts.get(article_type, f"{base_context}\n\nGenerate a comprehensive {article_type} article with all necessary legal provisions and subsections.")

class FallbackGenerator:
    """
    Provides fallback contract generation when LLM is unavailable or fails.
    Implements template-based generation with customization capabilities.
    """
    
    def __init__(self):
        """Initialize the fallback generator with standard templates."""
        self.article_templates = self._initialize_templates()
    
    def generate_fallback_article(self, article_type: str, requirements: ContractRequirements,
                                contract_data: GeneratedContractData) -> str:
        """
        Generate contract article using fallback templates when LLM fails.
        
        Args:
            article_type: Type of contract article to generate
            requirements: Contract requirements for customization
            contract_data: Contract data for personalization
            
        Returns:
            str: Generated article content using templates
        """
        template = self.article_templates.get(article_type, self._get_generic_template(article_type))
        logger.info(f"Using template for article: {article_type}")
        
        try:
            # Customize template with contract-specific data
            customized_content = template.format(
                contract_amount=contract_data.contract_amount,
                project_duration=contract_data.project_duration,
                retainage_percentage=contract_data.retainage_percentage,
                governing_law=contract_data.governing_law,
                liquidated_damages=max(500, requirements.contract_value // 10000),
                complexity=requirements.complexity
            )
        except Exception as e:
            logger.error(f"Error customizing template for {article_type}: {e}")
            customized_content = self._get_generic_template(article_type)
        
        return customized_content
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize standard contract article templates."""
        return {
            "Scope of Work and Specifications": """
ARTICLE - SCOPE OF WORK AND SPECIFICATIONS

1.1 GENERAL REQUIREMENTS
The Contractor shall provide all labor, materials, equipment, and services necessary for the complete execution of the project as specified in the contract documents.

1.2 WORK DESCRIPTION  
The work includes but is not limited to:
a) All activities specified in the project plans and specifications
b) Coordination with utilities and regulatory agencies
c) Compliance with all applicable codes and standards
d) Site preparation and cleanup activities

1.3 PERFORMANCE STANDARDS
All work shall be performed in accordance with:
a) Industry best practices and standards
b) Applicable building codes and regulations  
c) Project specifications and approved drawings
d) Quality control and inspection requirements

1.4 COMPLETION REQUIREMENTS
Work shall be deemed complete when all specified deliverables are finished and accepted by the Owner in accordance with contract terms.
""",

            "Contract Price and Payment Terms": """
ARTICLE - CONTRACT PRICE AND PAYMENT TERMS

2.1 CONTRACT PRICE
The total contract price is {contract_amount} for completion of all work specified in this contract.

2.2 PAYMENT SCHEDULE
Progress payments shall be made monthly based on work completed, less retainage of {retainage_percentage}.

2.3 INVOICE PROCEDURES
a) Contractor shall submit detailed monthly invoices
b) Payment shall be made within 30 days of approved invoice
c) All invoices must include supporting documentation

2.4 FINAL PAYMENT
Final payment shall be made upon substantial completion and acceptance of all work, release of liens, and satisfaction of warranty requirements.

2.5 RETAINAGE RELEASE
Retainage shall be released upon final completion and expiration of applicable warranty periods.
""",

            "Default and Termination": """
ARTICLE - DEFAULT AND TERMINATION

3.1 EVENTS OF DEFAULT
Default by Contractor includes but is not limited to:
a) Failure to commence work within specified time
b) Failure to maintain project schedule
c) Breach of contract terms and conditions
d) Insolvency or assignment for benefit of creditors

3.2 NOTICE AND CURE
Written notice of default shall be provided with opportunity to cure within 10 calendar days.

3.3 REMEDIES
Upon uncured default, Owner may:
a) Complete work using other contractors
b) Withhold payments due
c) Collect liquidated damages of ${liquidated_damages} per day of delay
d) Pursue all available legal remedies

3.4 TERMINATION
Either party may terminate this contract upon material breach by the other party, subject to notice and cure provisions.
"""
        }
    
    def _get_generic_template(self, article_type: str) -> str:
        """Provide generic template for unspecified article types."""
        return f"""
ARTICLE - {article_type.upper()}

This section contains standard provisions for the specified article. All terms and conditions are in accordance with applicable law and industry standards.

All parties agree to perform their obligations in good faith and in accordance with the contract terms, applicable law, and industry best practices.
"""

class SyntheticContractGenerator:
    """
    Main orchestrator class for synthetic construction contract generation.
    Coordinates all components and manages the complete generation workflow.
    """
    
    def __init__(self, api_key: Optional[str] = None, seed: Optional[int] = None):
        """
        Initialize the complete contract generation system.
        
        Args:
            api_key: Google Gemini API key for LLM generation
            seed: Random seed for reproducible synthetic data
        """
        # Initialize all component systems
        self.synthetic_generator = SyntheticDataGenerator(seed)
        self.content_validator = ContentValidator()
        self.llm_generator = LLMContentGenerator(api_key)
        self.fallback_generator = FallbackGenerator()
        
        # Track generation statistics
        self.generation_stats = {
            'total_generated': 0,
            'llm_successful': 0,
            'fallback_used': 0,
            'validation_issues': 0
        }
    
    def generate_complete_contract(self, requirements: ContractRequirements) -> Dict[str, Any]:
        logger.info(f"Starting contract generation for {requirements.project_type}")
        
        try:
            with st.status("Generating synthetic contract...", expanded=True) as status:
                status.write("Generating synthetic contract data...")
                contract_data = self.synthetic_generator.generate_contract_data(requirements)
                
                generated_articles = {}
                all_validation_issues = []
                
                for article_name in requirements.selected_articles:
                    status.write(f"Generating article: {article_name}")
                    article_content, issues = self._generate_single_article(
                        article_name, requirements, contract_data
                    )
                    generated_articles[article_name] = article_content
                    all_validation_issues.extend(issues)
                    
                    for issue in issues:
                        status.write(f"⚠️ {issue}")
                
                status.write("Assembling complete contract...")
                complete_contract = self._assemble_complete_contract(
                    generated_articles, contract_data, requirements
                )
                
                status.write("Validating and anonymizing final contract...")
                final_contract, final_issues = self.content_validator.validate_and_anonymize_content(
                    complete_contract
                )
                all_validation_issues.extend(final_issues)
                
                for issue in final_issues:
                    status.write(f"⚠️ {issue}")
                
                self.generation_stats['total_generated'] += 1
                if all_validation_issues:
                    self.generation_stats['validation_issues'] += 1
                
                status.write("Contract generation complete!")
                
                return {
                    'contract_text': final_contract,
                    'contract_data': asdict(contract_data),
                    'requirements': asdict(requirements),
                    'validation_issues': all_validation_issues,
                    'metadata': {
                        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'articles_generated': len(generated_articles),
                        'total_validation_issues': len(all_validation_issues),
                        'contract_length': len(final_contract.split()),
                        'generation_method': 'hybrid_llm_fallback'
                    },
                    'statistics': self.generation_stats.copy()
                }
                
        except Exception as e:
            logger.error(f"Contract generation failed: {e}")
            st.error(f"Contract generation failed: {str(e)}")
            raise Exception(f"Failed to generate contract: {str(e)}")
    
    def _generate_single_article(self, article_name: str, requirements: ContractRequirements,
                               contract_data: GeneratedContractData) -> Tuple[str, List[str]]:
        """
        Generate a single contract article with validation and anonymization.
        Attempts LLM generation first, falls back to templates if needed.
        
        Args:
            article_name: Name of article to generate
            requirements: Contract requirements for context
            contract_data: Synthetic contract data for personalization
            
        Returns:
            Tuple of (generated_content, validation_issues_found)
        """
        validation_issues = []
        raw_content = ""  # Initialize as empty string
        
        try:
            if self.llm_generator.model and self.llm_generator.is_initialized:
                logger.info(f"Generating {article_name} using LLM")
                raw_content = self.llm_generator.generate_article_content(
                    article_name, requirements, contract_data
                )
                self.generation_stats['llm_successful'] += 1
        except Exception as e:
            logger.warning(f"LLM generation failed for {article_name}: {e}")
            logger.info(f"Using fallback generation for {article_name}")
            try:
                raw_content = self.fallback_generator.generate_fallback_article(
                    article_name, requirements, contract_data
                )
                self.generation_stats['fallback_used'] += 1
            except Exception as fallback_e:
                logger.error(f"Fallback generation failed for {article_name}: {fallback_e}")
                raw_content = f"ARTICLE - {article_name}\n\n[Content generation failed. Please review and manually populate this section.]"
                validation_issues.append(f"Fallback generation failed for {article_name}")
        
        if not raw_content:
            raw_content = f"ARTICLE - {article_name}\n\n[No content generated. Please review and manually populate this section.]"
            validation_issues.append(f"No content generated for {article_name}")
        
        logger.info(f"Raw content for {article_name}: {raw_content[:100]}...")  # Log first 100 chars for debugging
        validated_content, issues = self.content_validator.validate_and_anonymize_content(raw_content)
        validation_issues.extend(issues)
        
        return validated_content, validation_issues
    
    def _assemble_complete_contract(self, articles: Dict[str, str], 
                                  contract_data: GeneratedContractData,
                                  requirements: ContractRequirements) -> str:
        """
        Assemble individual articles into a complete contract document.
        
        Args:
            articles: Dictionary of generated article content
            contract_data: Contract data for header information
            requirements: Contract requirements for context
            
        Returns:
            str: Complete assembled contract text
        """
        contract_parts = []
        
        # Contract header
        contract_parts.append(f"""
CONSTRUCTION CONTRACT

Contract Date: {contract_data.contract_date}
Project: {contract_data.project_name}
Contract Amount: {contract_data.contract_amount}

PARTIES TO THE CONTRACT:

OWNER: {contract_data.owner_name}
Representative: {contract_data.owner_representative}

CONTRACTOR: {contract_data.contractor_name}
Address: {contract_data.contractor_address}
Representative: {contract_data.contractor_representative}

PROJECT DESCRIPTION:
{contract_data.project_description}

""")
        
        # Add each generated article
        for article_name, content in articles.items():
            contract_parts.append(f"\n{'='*80}\n")
            contract_parts.append(f"{article_name.upper()}\n")
            contract_parts.append(f"{'='*80}\n")
            contract_parts.append(content)
            contract_parts.append("\n")
        
        return "\n".join(contract_parts)

def preprocess_markdown(text: str) -> str:
    """Preprocess contract text for better Markdown rendering."""
    replacements = {
        r'\[([A-Z_]+)\]': r'_\1_'  # Convert [PLACEHOLDER] to _PLACEHOLDER_
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def create_streamlit_interface():
    st.set_page_config(
        page_title="Synthetic Construction Contract Generator",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Enhanced Synthetic Construction Contract Generator")
    st.markdown("Generate realistic synthetic construction contracts with privacy protection")
    
    if 'generated_contract' not in st.session_state:
        st.session_state.generated_contract = None
    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False
    if 'selected_articles' not in st.session_state:
        st.session_state.selected_articles = []
    
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google Gemini API Key (Optional)", 
            type="password",
            help="Enter a valid Google Gemini API key for LLM generation. Leave empty to use template-based generation."
        )
        
        use_seed = st.checkbox("Use Random Seed for Reproducibility")
        seed_value = None
        if use_seed:
            seed_value = st.number_input("Random Seed", value=42, min_value=1, max_value=10000)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Contract Setup", "Generated Contract", "Analytics", "Markdown Preview"])
    
    with tab1:
        st.header("Contract Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            project_type = st.selectbox(
                "Project Type",
                [
                    "Highway Construction and Reconstruction",
                    "Bridge Design and Construction",
                    "Municipal Water Treatment Facility", 
                    "Educational Institution Building",
                    "Public Safety Complex",
                    "Airport Infrastructure Development",
                    "Transit System Expansion",
                    "Environmental Remediation Project",
                    "Energy Infrastructure Installation",
                    "Commercial Office Building"
                ]
            )
            
            contract_value = st.number_input(
                "Contract Value ($)",
                min_value=50000,
                max_value=50000000,
                value=1000000,
                step=50000,
                format="%d"
            )
            
            complexity_options = ["simple", "moderate", "complex", "mega"]
            complexity_index = st.slider(
                "Project Complexity (slide to select)",
                min_value=0,
                max_value=len(complexity_options)-1,
                value=1
            )
            complexity = complexity_options[complexity_index]
            st.write(f"Selected Complexity: **{complexity}**")
        
        with col2:
            duration_days = st.number_input(
                "Project Duration (Days)",
                min_value=30,
                max_value=1095,
                value=365,
                step=30
            )
            
            location = st.text_input(
                "Project Location",
                value="Metropolitan Area Alpha"
            )
        
        st.subheader("Select Contract Articles to Generate")
        
        available_articles = [
            "Definitions and Interpretation",
            "Scope of Work and Specifications", 
            "Contract Price and Payment Terms",
            "Time for Performance and Delays",
            "Change Orders and Modifications",
            "Quality Control and Inspections",
            "Insurance and Bonding Requirements",
            "Safety and Environmental Compliance",
            "Default and Termination",
            "Dispute Resolution",
            "General Provisions",
            "Signature Blocks"
        ]
        
        col1, col2, col3 = st.columns(3)
        articles_per_column = len(available_articles) // 3
        
        for i, article in enumerate(available_articles):
            col = col1 if i < articles_per_column else col2 if i < 2 * articles_per_column else col3
            with col:
                is_checked = article in st.session_state.selected_articles
                if st.checkbox(article, value=is_checked, key=f"article_{article}"):
                    if article not in st.session_state.selected_articles:
                        st.session_state.selected_articles.append(article)
                else:
                    if article in st.session_state.selected_articles:
                        st.session_state.selected_articles.remove(article)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Select Common Articles"):
                common_articles = [
                    "Scope of Work and Specifications",
                    "Contract Price and Payment Terms", 
                    "Insurance and Bonding Requirements",
                    "Default and Termination"
                ]
                st.session_state.selected_articles = common_articles
                st.rerun()
        
        with col2:
            if st.button("Select All Articles"):
                st.session_state.selected_articles = available_articles.copy()
                st.rerun()
        
        with col3:
            if st.button("Clear All Selections"):
                st.session_state.selected_articles = []
                st.rerun()
        
        st.subheader("Additional Requirements")
        special_requirements = st.text_area(
            "Special Requirements (one per line)",
            placeholder="Environmental impact assessment\nHistoric preservation compliance\nMinority business participation"
        ).split('\n') if st.text_area(
            "Special Requirements (one per line)",
            placeholder="Environmental impact assessment\nHistoric preservation compliance\nMinority business participation"
        ) else []
        
        if st.button("Generate Synthetic Contract", type="primary", use_container_width=True):
            if not st.session_state.selected_articles:
                st.error("Please select at least one contract article to generate.")
            else:
                with st.spinner("Initializing contract generation..."):
                    try:
                        requirements = ContractRequirements(
                            project_type=project_type,
                            contract_value=contract_value,
                            duration_days=duration_days,
                            location=location,
                            complexity=complexity,
                            selected_articles=st.session_state.selected_articles,
                            special_requirements=[req.strip() for req in special_requirements if req.strip()]
                        )
                        
                        generator = SyntheticContractGenerator(
                            api_key=api_key if api_key else None,
                            seed=seed_value
                        )
                        
                        if api_key and not generator.llm_generator.is_initialized:
                            st.warning("LLM initialization failed. Using template-based generation. Please verify your Google Gemini API key.")
                        
                        result = generator.generate_complete_contract(requirements)
                        
                        st.session_state.generated_contract = result
                        st.session_state.generation_complete = True
                        
                        st.success(f"Contract generated successfully! Generated {len(st.session_state.selected_articles)} articles.")
                        st.info("Check the 'Generated Contract' or 'Markdown Preview' tabs to view your contract.")
                        
                    except Exception as e:
                        st.error(f"Error generating contract: {str(e)}")
                        logger.error(f"Contract generation error: {e}")
    
    with tab2:
        st.header("Generated Contract")
        
        if st.session_state.generation_complete and st.session_state.generated_contract:
            result = st.session_state.generated_contract
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Articles Generated", result['metadata']['articles_generated'])
            with col2:
                st.metric("Contract Length", f"{result['metadata']['contract_length']} words")
            with col3:
                st.metric("Validation Issues", result['metadata']['total_validation_issues'])
            
            if result['validation_issues']:
                with st.expander("Validation Issues Found", expanded=False):
                    for issue in result['validation_issues']:
                        st.warning(issue)
            
            st.subheader("Contract Content")
            contract_text = result['contract_text']
            
            edited_contract = st.text_area(
                "Generated Contract (Editable)",
                value=contract_text,
                height=600,
                help="You can edit the contract text here before downloading"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Contract (.txt)",
                    data=edited_contract,
                    file_name=f"synthetic_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                json_data = {
                    'contract_text': edited_contract,
                    'metadata': result['metadata'],
                    'contract_data': result['contract_data']
                }
                st.download_button(
                    label="Download Full Data (.json)",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"contract_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No contract generated yet. Use the 'Contract Setup' tab to create a contract.")
    
    with tab3:
        st.header("Generation Analytics")
        
        if st.session_state.generation_complete and st.session_state.generated_contract:
            result = st.session_state.generated_contract
            stats = result['statistics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Generation Statistics")
                st.write(f"**Total Contracts Generated:** {stats['total_generated']}")
                st.write(f"**LLM Successful:** {stats['llm_successful']}")
                st.write(f"**Fallback Used:** {stats['fallback_used']}")
                st.write(f"**Validation Issues:** {stats['validation_issues']}")
            
            with col2:
                st.subheader("Contract Details")
                metadata = result['metadata']
                st.write(f"**Generation Date:** {metadata['generation_date']}")
                st.write(f"**Generation Method:** {metadata['generation_method']}")
                st.write(f"**Articles Count:** {metadata['articles_generated']}")
                st.write(f"**Word Count:** {metadata['contract_length']}")
            
            st.subheader("Synthetic Contract Data")
            contract_data = result['contract_data']
            
            info_data = {
                'Field': [
                    'Project Name', 'Contract Amount', 'Duration', 
                    'Start Date', 'Completion Date', 'Contract Type'
                ],
                'Value': [
                    contract_data['project_name'],
                    contract_data['contract_amount'], 
                    contract_data['project_duration'],
                    contract_data['start_date'],
                    contract_data['completion_date'],
                    contract_data['contract_type']
                ]
            }
            
            df = pd.DataFrame(info_data)
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("Generate a contract first to see analytics.")
    
    with tab4:
        st.header("Markdown Preview")
        
        if st.session_state.generation_complete and st.session_state.generated_contract:
            contract_text = preprocess_markdown(st.session_state.generated_contract['contract_text'])
            st.markdown(contract_text, unsafe_allow_html=False)
            
            st.download_button(
                label="Download Markdown (.md)",
                data=contract_text,
                file_name=f"synthetic_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("No contract generated yet. Use the 'Contract Setup' tab to create a contract.")

if __name__ == "__main__":
    try:
        create_streamlit_interface()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Streamlit application error: {e}")