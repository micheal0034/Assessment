# Neo4j GraphRAG for Cybersecurity Knowledge

A powerful system for building and querying knowledge graphs from NIST cybersecurity documentation using Neo4j, vector embeddings, and RAG (Retrieval-Augmented Generation).

## Overview

This system combines several advanced technologies to provide intelligent querying of NIST cybersecurity standards and guidelines:

1. **Knowledge Graph Construction**: Automatically extracts entities and relationships from NIST cybersecurity documents to build a structured knowledge graph in Neo4j.
2. **Vector Embeddings**: Implements semantic search capabilities through vector embeddings for accurate retrieval.
3. **Retrieval-Augmented Generation (RAG)**: Enhances LLM responses by providing relevant context from the knowledge graph.

## Requirements

- Python 3.10+
- Neo4j database instance
- Access to embedding models (Ollama -- llama vision or deepseek)
- NIST cybersecurity documents in the specified directory

### Python Dependencies

```
neo4j
asyncio
colorama
neo4j_graphrag
rich
langchain_google_genai
python-decouple
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file with the following parameters:

```
NEO4J_URI=""
NEO4J_USERNAME=""
NEO4J_PASSWORD=""
```

## Key Components

### 1. Knowledge Graph Generation

The system can build a knowledge graph from NIST cybersecurity documents using LLM-based extraction:

- Processes PDF documents using a chunking strategy
- Extracts entities and relationships with customizable schemas
- Stores the structured data in Neo4j

### 2. Vector Embedding and Retrieval

Vector embeddings enable semantic search across the knowledge base:

- Uses Ollama embeddings (default model: `llama3.2-vision`and `tazarov/all-minilm-l6-v2-f32`)
- Creates and manages vector indexes in Neo4j
- Implements similarity search for relevant content retrieval

### 3. RAG Implementation

The RAG system enhances question answering:

- Retrieves relevant context from the knowledge graph
- Formats prompts with the cybersecurity expert template
- Generates accurate responses based on retrieved information

## Usage

### Basic Setup

1. Ensure Neo4j is running and accessible with the configured credentials
2. Place NIST cybersecurity documents in the `../../data/NIST-Cybersecurity-Documents` directory
3. Run the script to initialize the system

### Knowledge Graph Construction

Set `CREATE_KG_GRAPH = True` to build or update the knowledge graph. This process:

- Chunks documents into manageable segments
- Extracts entities and relationships
- Populates the Neo4j database

`Because of some issues, i have to upload the documents in neo4j llm graph builder -- done in split seconds.....`

### Vector Index Creation

Set `INIT_INDEX = True` to create or recreate the vector index. Only needed during initial setup or when changing embedding dimensions.

### Query Execution

The system comes with a default query to list contributors to NIST documents, but you can modify the `user_prompt` variable to ask any cybersecurity-related question.

## Parameters

### System Configuration

- `CREATE_KG_GRAPH`: Toggle knowledge graph creation (default: False)
- `INIT_INDEX`: Toggle vector index creation (default: False)
- `vector_index_name`: Name of the vector index in Neo4j (default: "vector")

### Text Processing

- `chunk_size`: Size of text chunks for processing (default: 200)
- `chunk_overlap`: Overlap between chunks to maintain context (default: 20)

### Generation Parameters

- `TEMPERATURE`: Controls randomness in LLM responses (default: 0.5)
- `TOP_P`: Controls diversity in LLM responses (default: 0.9)

## Customization

### Schema Definition

Customize the knowledge graph structure by modifying the `create_schema_definition()` function:

- Define node types for different entity categories
- Define relationship types to connect entities

### RAG Template

Customize the prompt template in `rag_template_text` to adjust how the system responds to queries.

## Troubleshooting

### Vector Index Issues

If you encounter a "no index with name X found" error:
- Ensure the index name in `vector_index_name` matches what's in your Neo4j database
- Run with `INIT_INDEX = True` to create the index if it doesn't exist

### Embedding Model Errors

If embedding generation fails:
- Verify the model name is correct and accessible
- Check that Ollama is properly configured and running

### Document Processing Failures

If document processing fails:
- Verify the document path is correct
- Ensure documents are in the expected format (PDF)
- Check console output for specific error messages

## Advanced Usage

### Custom Document Sets

Modify the `files` list in the `generate_knowledge_graph()` function to process specific documents:

```python
files = ["SP800-21-1_Dec2005", "SP800-57part1rev4", "SP800-57part2"]
```

### Adjusting Response Format

Modify the `postprocess_rag_completion()` function to customize how responses are processed and formatted.

## Example Query and Response

Input:
```
Return a numbered list of all individual authors/contributors to the NIST cybersecurity documents.
```

The system will:
1. Embed the query
2. Retrieve relevant context from the knowledge graph
3. Generate a response that lists the authors/contributors based on the information in the NIST documents





