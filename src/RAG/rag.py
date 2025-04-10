import os
import asyncio
import json
import neo4j

from colorama import Fore

from neo4j_graphrag.llm import OllamaLLM, OpenAILLM
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from rich import print

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from decouple import config
from rich import print

print("Starting execution...")

NEO4J_URI = config("AURA_NEO4J_CONNECTION_URI")
NEO4J_USERNAME = config("NEO4J_USERNAME")
NEO4J_PASSWORD = config("NEO4J_PASSWORD")
CREATE_KG_GRAPH = False
INIT_INDEX = False

print(f"Configuration loaded: URI={NEO4J_URI}, Username={NEO4J_USERNAME}, Create KG={CREATE_KG_GRAPH}, Init Index={INIT_INDEX}")

try:
    neo4j_driver = neo4j.GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    print("Neo4j connection established successfully")
except Exception as e:
    print(f"Error connecting to Neo4j: {str(e)}")
    raise

model_name = "tazarov/all-minilm-l6-v2-f32"
document_path = "../data/nist_cybersecurity_documents"
print(f"Using model: {model_name}")
print(f"Document path: {document_path}")
try:
    files_in_dir = os.listdir(document_path)
    print(f"Files in document directory: {files_in_dir}")
except Exception as e:
    print(f"Error listing documents directory: {str(e)}")

vector_index_name = "vector"
chunk_size = 200
chunk_overlap = 20
TOP_P = 0.9
TEMPERATURE = 0.5

print(f"Vector index: {vector_index_name}, Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
print(f"Generation params: Temperature={TEMPERATURE}, Top_p={TOP_P}")


def create_schema_definition():
    print("Defining knowledge graph schema...")
    # define node labels
    basic_node_labels = ["Object", "Entity", "Group", "Person", "OrganizationOrInstitution", "IdeaOrConcept", "GeographicLocation"]

    academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

    node_labels = basic_node_labels + academic_node_labels

    # define relationship types
    relation_types = [
        "CREATED_BY", "IMPACTS", "EVALUATES", "RELATED_TO", "WROTE",
        "RESULTS_IN", "SUPPORTS", "DEFINES", "RECOMMENDS",
        "CONTAINS", "COMMUNICATES_WITH", "SPECIFIES", "GENERATES", "OBTAINED", "APPLIED_TO"
    ]
    
    print(f"Generated {len(node_labels)} node labels and {len(relation_types)} relationship types")
    return node_labels, relation_types


async def generate_knowledge_graph(
    path,
    driver,
    embedder,
    llm,
    chunk_size=500,
    chunk_overlap=100,
    generate_schema=False
):
    print(f"Starting knowledge graph generation with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    prompt_template = '''
    As a cybersecurity expert specialized in knowledge extraction, your task is to convert the provided text 
    into a structured property graph that captures cybersecurity concepts, entities, and their relationships.

    Extract the entities (nodes) and specify their type from the following Input text.
    Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
      "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

    - Use only the information from the Input text. Do not add any additional information.  
    - If the input text is empty, return an empty JSON. 
    - Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
    - An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
    - Multiple documents may be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 

    Use only fhe following nodes and relationships (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Do not return any additional information other than the JSON in it.
    '''
    if generate_schema:
        print("Generating schema for KG...")
        entities, relations = create_schema_definition()
    else:
        print("Using default schema for KG")
        entities, relations = None, None

    print("Initializing KG pipeline...")
    kg_graph = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        prompt_template=prompt_template,
        entities=entities,
        relations=relations,
        text_splitter=FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        from_pdf=True
    )

    print(f"Current working directory: {os.getcwd()}")
    files = ["SP800-21-1_Dec2005","SP800-57part1rev4","SP800-57part2"]
    print(f"Files to process: {files}")
    fpaths = [os.path.join(path, f) for f in files]
    for fpath in fpaths:
        print(f"Processing {fpath}")
        pdf_result = await kg_graph.run_async(file_path=fpath)
        print(f"PDF processing result: {pdf_result}\n\n")

    print("Knowledge graph generation completed")
    return kg_graph

def generate_vector_retriever(driver, embedder=None, dimensions=3584, index_name="text_embeddings", init_index=False):
    print(f"Generating vector retriever with dimensions={dimensions}, index_name={index_name}, init_index={init_index}")
    if init_index:
        print(f"Creating vector index '{index_name}' with {dimensions} dimensions")
        try:
            create_vector_index(
                driver,
                name=index_name,
                label="Chunk",
                embedding_property="embedding",
                dimensions=dimensions,
                similarity_fn="cosine"
            )
            print("Vector index created successfully")
        except Exception as e:
            print(f"Error creating vector index: {str(e)}")
    
    print("Initializing vector retriever...")
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=index_name,
        embedder=embedder,
        return_properties=["text"]
    )
    print("Vector retriever initialized")
    return vector_retriever


#  Instantiate embedder_llm
print("Initializing embedder LLM...")
try:
    embedder_llm = OpenAILLM(
        model_name="deepseek/deepseek-v3-base:free",
        api_key=config("Deepseek_API"),
        base_url=config("OPENROUTER_base_url"),
        model_params={
            "response_format": {"type": "json_object"},
            "temperature": TEMPERATURE,
            "top_p": TOP_P
        },
    )
    print("Embedder LLM initialized successfully")
except Exception as e:
    print(f"Error initializing embedder LLM: {str(e)}")
    raise


#  Instantiate embedder
print("Initializing embedder...")
try:
    embedder = OllamaEmbeddings(model="tazarov/all-minilm-l6-v2-f32")
    print("Embedder initialized successfully")
except Exception as e:
    print(f"Error initializing embedder: {str(e)}")
    raise

print("Testing embedder with a sample query...")
try:
    num_dimensions = len(embedder.embed_query("Who is Ozymandias?"))
    print(f"Embedder test successful. Vector dimensions: {num_dimensions}")
except Exception as e:
    print(f"Error testing embedder: {str(e)}")
    raise

#  Create knowledge graph, if needed
if CREATE_KG_GRAPH:
    print("Creating knowledge graph...")
    try:
        kg_graph = generate_knowledge_graph(
            path=document_path,
            embedder=embedder,
            llm=embedder_llm,
            driver=neo4j_driver,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        kg_graph = asyncio.run(kg_graph)
        print("Knowledge graph created successfully")
    except Exception as e:
        print(f"Error creating knowledge graph: {str(e)}")
        raise

#  Instantiate vector retriever
print("Setting up vector retriever...")
try:
    vector_retriever = generate_vector_retriever(
        driver=neo4j_driver,
        index_name=vector_index_name,
        embedder=embedder,
        dimensions=num_dimensions,
        init_index=INIT_INDEX
    )
    print("Vector retriever setup complete")
except Exception as e:
    print(f"Error setting up vector retriever: {str(e)}")
    raise

#  Instantiate llm for RAG
# 3. GraphRAG Class
print("Initializing LLM for RAG...")
try:
    llm = OllamaLLM(
        model_name="llama3.2-vision",
        model_params={
           "temperature": TEMPERATURE,
           "top_p": TOP_P,
       }
    )
    print("LLM for RAG initialized successfully")
except Exception as e:
    print(f"Error initializing LLM for RAG: {str(e)}")
    raise

#  Instantiate RAG text template
print("Creating RAG template...")
rag_template_text = '''
You are an expert cybersecurity analyst with deep knowledge of NIST standards and frameworks.

Your task is to provide accurate, informative answers to questions about cybersecurity based on the provided Context.

Guidelines:
- Base your responses solely on the information in the Context
- Provide detailed explanations with supporting evidence from the Context
- Be precise and technical in your analysis
- If the Context doesn't contain relevant information, acknowledge the limitations
- Do not fabricate information beyond what's provided
# Question:
{query_text}

# Context:
{context}

# Answer:
'''
rag_template = RagTemplate(template=rag_template_text, expected_inputs=['query_text', 'context'])
print("RAG template created")

#  Instantiate GraphRAG instance
print("Initializing GraphRAG...")
try:
    rag = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)
    print("GraphRAG initialized successfully")
except Exception as e:
    print(f"Error initializing GraphRAG: {str(e)}")
    raise

def extract_json_from_content(content):
    print(f"Extracting JSON from: {content[:100]}...")
    try:
        start_index = content.index("{")
        end_index = content.index("}")
        extracted_dict = json.loads(content[start_index : end_index + 1])
        print(f"JSON extracted successfully with {len(extracted_dict)} keys")
        return extracted_dict
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}")
        raise

def postprocess_rag_completion(completion):
    try:
        if "</think>" in completion.answer:
            processed = completion.answer.split("</think>")[-1].strip()
        else:
            processed = completion.answer
        print(f"Processed completion: {processed[:100]}...")
        return processed
    except Exception as e:
        print(f"Error in postprocessing: {str(e)}")
        # Fallback to returning the original completion if error occurs
        return completion.answer

# 4. Run

if __name__ == "__main__":
    user_prompt = '''
    Perform a detailed analysis of the key components of the NIST cybersecurity documents.
    '''
    print(f"Processing user prompt: '{user_prompt}'")
    completion = rag.search(user_prompt)
    print("Performing RAG search...")
    try:
        response = postprocess_rag_completion(completion)
        print(f"Postprocessed response: {response}")
    except Exception as e:
        print(f"Error in RAG search: {str(e)}")
        response = "An error occurred during search."

    print(Fore.BLUE + ">>> User")
    print(Fore.BLUE + f"{user_prompt}\n\n")

    print(Fore.MAGENTA + ">>> Response")
    print(Fore.MAGENTA + f"{response}\n\n" + Fore.RESET)

    # Display the retrieved results for debugging
    print("Getting search results from vector retriever...")
    try:
        vector_res = vector_retriever.get_search_results(
            query_text=user_prompt,
            top_k=3
        )
        print(f"Retrieved {len(vector_res.records) if vector_res and hasattr(vector_res, 'records') else 0} records")
        
        if vector_res and hasattr(vector_res, 'records'):
            for i, record in enumerate(vector_res.records):
                print(f"==== RETRIEVED RESULT {i+1} ====")
                try:
                    data = record.data()
                    text = data["node"]["text"]
                    print(text)
                except Exception as e:
                    print(f"Error displaying record: {str(e)}")
        else:
            print("No records retrieved or invalid response format")
    except Exception as e:
        print(f"Error retrieving vector results: {str(e)}")


