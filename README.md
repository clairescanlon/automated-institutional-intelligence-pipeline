# Automated Institutional Intelligence Pipeline
End-to-end data pipeline transforming unstructured text-heavy PDFs into a mapped knowledge base of entities, links, and operational directives. 

> [!NOTE]
> This project is currently being updated. Check back soon for updates.


## Overview
Automated intelligence engine transforming dense PDFs into a structured relational database. The system bypasses complex rhetorical framing to expose the underlying operational logic and dependencies. This allows for the discovery of hidden patterns, risks, and operational shifts that are otherwise obscured.

## Architecture
The data pipeline utilizes a modular serverless architecture on AWS.
1. **Ingestion Layer:** Programmatically retrieves high-volume directives via REST APIs and specialized web-scraping modules. 
2. **Transformation Layer:** AWS Lambda functions handle text extraction, OCR, and structural parsing. This layer converts binary PDF data into machine-readable text while preserving metadata hierarchies and complex document layouts.
3. **Validation & Integrity Layer:** Rigorous runtime validation using Pandera and Pydantic. This layer enforces schema constraints and data-type consistency, preventing bad data from reaching production tables.
4. **Persistence Layer:**  Fully processed records are committed to an Aurora PostgreSQL relational database.
5. **Observability & Monitoring Layer:**  Real-time health tracking and forensic logging via AWS CloudWatch and Loguru. This layer integrates AWS CloudTrail for API auditing, AWS Config for infrastructure compliance, and Amazon SNS for automated alerting on schema violations or ingestion failures.

## Key Features
* **Automated Document Ingestion:** Programmatically retrieves high-volume directives via APIs and web scraping, storing raw assets in AWS S3 for downstream processing.
* **Rhetoric Neutralization:** Isolates core operational logic from subjective framing using SVO extraction, quantifying complexity to identify and bypass obfuscated content.
* **Systemic Entity Mapping:**  A sophisticated entity resolution module that identifies and links disparate stakeholders and business units.
* **Hybrid Extraction & OCR:**  A dual-layer text recovery suite combining native PDF parsing with Tesseract-based OCR, ensuring 100% data fidelity from both digital-native assets and legacy image-based files while preserving complex metadata.
* **Automated Data Integrity & Validation:** Rigorous runtime validation using Pandera and Pydantic to enforce schema constraints at every pipeline stage.
* **Systemic Risk Mapping**: Utilizes graph-based analysis to uncover the cumulative impacts of localized operational shifts.

## Tech Stack
* Amazon Web Services (AWS)
* Python
* Amazon Aurora PostgreSQL
* Tesseract OCR

## Core Tables

| Table | Description |
|---|---|
| `source_documents` | Each ingested document |
| `domains` | Domain affected |
| `stated_outcomes` | What the document claims it does |
| `actual_outcomes` | What the analysis determined it actually does |
| `losers` | Who it benefitted |
| `winners` | Who it harmed |
| `actions` | Concrete action taken |
| `marker_inversions` | The rhetoric-to-reality gap — links a marker to its stated vs. actual meaning |
| `rhetorical_markers` | The individual keywords and phrases being tracked |
| `entities` | People, teams, organizations mentioned |

