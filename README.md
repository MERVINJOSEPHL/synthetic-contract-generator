# Synthetic Construction Contract Data Generator

**Mervin Joseph L**  
**Version 1.0.0 - June 2025**

This project is a Proof-of-Concept (PoC) for generating synthetic construction contract data, focusing on privacy-preserving, realistic contract clauses. The system leverages an LLM (Google Gemini) for content generation, Faker for synthetic entities, and a custom anonymization layer to ensure privacy. The primary focus is on generating a "Scope of Work" clause, with scalability and extensibility in mind.

## Frameworks and Libraries Used
- **Python**: Core programming language for implementation
- **Streamlit**: Interactive UI for contract generation and visualization
- **Google Gemini API**: LLM for generating contract clause content
- **Faker**: For creating realistic synthetic entities (e.g., names, locations, dates)
- **Pandas**: Data handling and analytics display
- **Logging**: For error tracking and debugging
- **Regex**: For content validation and anonymization

## Project Objectives
- **Privacy Preservation**: Ensure no real-world sensitive data is included, using anonymization techniques (e.g., replacing identifiable info with placeholders like `[PROJECT_NAME]`, `[CONTRACTOR_NAME]`).
- **Fidelity to Real Data**: Generate contract clauses that mimic real-world construction contract structure, vocabulary (e.g., "installation", "materials", "subcontractor"), and relationships.
- **Scalability (Conceptual)**: Design a system that can scale to generate multiple contract articles and handle larger datasets.

## Core Functionality
The system focuses on generating a synthetic "Scope of Work" clause for a construction contract. Key features include:
- **Customization**: Users can specify project type, contract value, duration, location, and complexity via a Streamlit interface.
- **LLM-Powered Generation**: Utilizes Google Gemini API for generating realistic contract clause content (with fallback templates if the API fails).
- **Anonymization**: A `ContentValidator` class scans and replaces sensitive data (e.g., phone numbers, emails, real company names) with placeholders.
- **Validation and Post-Processing**: Ensures generated content includes construction-specific keywords and adheres to a structured format (e.g., numbered sections).
- **Output**: Displays the generated contract in a multi-tabbed Streamlit UI with options to view, edit, download (as TXT or Markdown), and analyze generation stats.

## System Flow (Based on Diagram)
1. **User Input**: The user provides project details (type, duration, value, complexity, location) via a Streamlit form.
2. **Synthetic Data Generation**: The `SyntheticDataGenerator` class uses Faker to create realistic project details (e.g., project name, contractor/owner names, dates).
3. **LLM or Template Generation**:
   - If a Google Gemini API key is provided, the `LLMContentGenerator` attempts to generate the "Scope of Work" clause.
   - If the API is unavailable or fails, the `FallbackGenerator` uses predefined templates to generate the clause.
4. **Content Validation**:
   - The `ContentValidator` checks for sensitive data (e.g., PII, real company names) and anonymizes it.
   - Additional checks ensure construction-specific context (e.g., no real brand names like "Caterpillar", inclusion of relevant terms).
5. **Assembly and Output**:
   - The full contract is assembled with a header, table of contents, and the generated clause.
   - The contract is displayed in Streamlit with tabs for viewing, editing, analytics, and Markdown preview.
6. **Export and Deployment**:
   - Users can download the contract as a TXT or Markdown file.
   - The system is deployed and accessible via a public URL.

## Code Quality
- **Modularity**: Classes like `SyntheticDataGenerator`, `ContentValidator`, `LLMContentGenerator`, and `FallbackGenerator` ensure separations of concerns.
- **Error Handling**: Robust logging and exception handling for API calls, validation, and generation steps.
- **Documentation**: Comprehensive docstrings and comments explain functionality and intent.
- **Dependencies**: Listed in `requirements.txt` (e.g., streamlit, faker, pandas, google-generativeai).

## Deployed URL
The application is deployed and accessible at:  
[https://synthetic-contract-generator.onrender.com/]

## Conclusion
The Synthetic Construction Contract Data Generator successfully delivers a privacy-preserving, realistic "Scope of Work" clause for construction contracts. It combines LLM-powered generation with robust validation with a user-friendly Streamlit interface, laying the foundation for a scalable synthetic data solution in the construction industry.
