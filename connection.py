from neo4j import GraphDatabase
from decouple import config


# Get Neo4j credentials from .env file
uri = config('NEO4J_URI')
username = config('NEO4J_USERNAME')
password = config('NEO4J_PASSWORD')

# Establish a connection to the database
def test_connection():
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Neo4j!' AS message")
            for record in result:
                print(record['message'])
        print("Connection successful!")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

# Run the connection test
test_connection()
