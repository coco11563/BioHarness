"""Extraction prompts for biomedical KG construction.

These prompts are optimized for:
- Biomedical text (scientific papers, abstracts)
- JSON output format (cleaner than delimiter-based)
- Entity types specific to biomedicine
"""

BIOMEDICAL_ENTITY_TYPES = [
    "gene",
    "protein",
    "disease",
    "drug",
    "chemical",
    "pathway",
    "cell_type",
    "organism",
    "anatomy",
    "biological_process",
    "method",
    "concept",
    "other",
]

BIOMEDICAL_RELATION_TYPES = [
    "treats",
    "causes",
    "associated_with",
    "inhibits",
    "activates",
    "regulates",
    "expressed_in",
    "interacts_with",
    "part_of",
    "located_in",
    "related_to",
    "other",
]

ENTITY_EXTRACTION_PROMPT = """You are a biomedical knowledge graph specialist. Extract entities from the given text.

## Entity Types
{entity_types}

## Instructions
1. Identify all biomedical entities (genes, proteins, diseases, drugs, pathways, etc.)
2. For each entity, provide:
   - name: Canonical name (use official nomenclature when possible)
   - type: One of the entity types listed above
   - description: Brief description based on the text context
3. Normalize entity names:
   - Use official gene symbols (e.g., "TP53" not "p53 tumor suppressor")
   - Use generic drug names (e.g., "metformin" not "Glucophage")
   - Use MeSH-style disease names when possible

## Output Format
Return a JSON array of entities:
```json
[
  {{"name": "entity_name", "type": "entity_type", "description": "description from text"}}
]
```

## Text to Process
{text}

## Output
Return ONLY the JSON array, no additional text:"""


RELATION_EXTRACTION_PROMPT = """You are a biomedical knowledge graph specialist. Extract relationships between entities.

## Given Entities
{entities}

## Relation Types
{relation_types}

## Instructions
1. Identify relationships between the given entities based on the text
2. For each relationship, provide:
   - source: Source entity name (must match an entity from the list)
   - target: Target entity name (must match an entity from the list)
   - type: Relationship type from the list above
   - description: Brief description of the relationship
   - keywords: 2-3 keywords describing the relationship
3. Only extract relationships that are explicitly stated or strongly implied in the text
4. Prefer specific relation types over "related_to" when possible

## Output Format
Return a JSON array of relationships:
```json
[
  {{
    "source": "entity1",
    "target": "entity2",
    "type": "relation_type",
    "description": "description of relationship",
    "keywords": ["keyword1", "keyword2"]
  }}
]
```

## Text to Process
{text}

## Output
Return ONLY the JSON array, no additional text:"""


ENTITY_DESCRIPTION_MERGE_PROMPT = """You are a knowledge graph curator. Merge multiple descriptions of the same entity.

## Entity Name
{entity_name}

## Entity Type
{entity_type}

## Descriptions
{descriptions}

## Instructions
1. Synthesize all descriptions into a single comprehensive summary
2. Remove redundant information
3. Preserve all unique facts and relationships mentioned
4. Keep the summary concise (max 200 words)
5. Write in third person, objective tone

## Output
Return ONLY the merged description text, no JSON or formatting:"""


COMMUNITY_SUMMARY_PROMPT = """You are a biomedical knowledge synthesizer. Summarize this community of related entities.

## Community Entities
{entities}

## Community Relations
{relations}

## Instructions
1. Provide a concise title (5-10 words) capturing the main theme
2. Write a summary (100-200 words) describing:
   - The main topic or theme of this entity cluster
   - Key entities and their roles
   - Important relationships and patterns
3. Focus on biomedical significance and clinical relevance

## Output Format
Return JSON:
```json
{{
  "title": "Community title",
  "summary": "Comprehensive summary of the community..."
}}
```

## Output
Return ONLY the JSON, no additional text:"""


KEYWORDS_EXTRACTION_PROMPT = """Extract search keywords from this query for biomedical knowledge graph retrieval.

## Query
{query}

## Instructions
Extract two types of keywords:
1. high_level_keywords: Broad concepts, themes, or question types
2. low_level_keywords: Specific entities, technical terms, proper nouns

## Output Format
```json
{{
  "high_level_keywords": ["keyword1", "keyword2"],
  "low_level_keywords": ["entity1", "term1", "name1"]
}}
```

## Output
Return ONLY the JSON, no additional text:"""
