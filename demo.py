import ollama
import os

# Configuration
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

# Global vector database
VECTOR_DB = []

def load_dataset(filename):
    """Load and chunk the knowledge base from a text file."""
    with open(filename, "r", encoding="utf-8") as file:
        dataset = [line.strip() for line in file if line.strip()]
    return dataset

def add_chunk_to_database(chunk):
    """Request an embedding and append (chunk, embedding) to VECTOR_DB."""
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))

def cosine_similarity(a, b):
    """Return the cosine similarity between vectors a and b."""
    # Handle zero-length vectors
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    dot_product = sum(x*y for x, y in zip(a, b))
    return dot_product / (norm_a * norm_b)

def test_cosine_similarity():
    """Test the cosine similarity function with known values."""
    print("\n=== Cosine Similarity Tests ===")
    # Test 1: Identical vectors should have similarity 1.0
    v1 = [1, 0, 0]
    v2 = [1, 0, 0]
    result1 = cosine_similarity(v1, v2)
    print(f"Test 1 (identical vectors): {result1} (expected: 1.0)")
    assert abs(result1 - 1.0) < 0.001, "Test 1 failed"
    
    # Test 2: Orthogonal vectors should have similarity 0.0
    v3 = [1, 0]
    v4 = [0, 1]
    result2 = cosine_similarity(v3, v4)
    print(f"Test 2 (orthogonal vectors): {result2} (expected: 0.0)")
    assert abs(result2 - 0.0) < 0.001, "Test 2 failed"
    
    # Test 3: Opposite vectors should have similarity -1.0
    v5 = [1, 1]
    v6 = [-1, -1]
    result3 = cosine_similarity(v5, v6)
    print(f"Test 3 (opposite vectors): {result3} (expected: -1.0)")
    assert abs(result3 - (-1.0)) < 0.001, "Test 3 failed"
    
    # Test 4: Zero vector should return 0.0
    v7 = [0, 0]
    v8 = [1, 1]
    result4 = cosine_similarity(v7, v8)
    print(f"Test 4 (zero vector): {result4} (expected: 0.0)")
    assert result4 == 0.0, "Test 4 failed"
    
    print("✅ All cosine similarity tests passed!")

def retrieve(query, top_n=3):
    """Retrieve the top_n chunks most similar to the query."""
    # Defensive check for empty database
    if not VECTOR_DB:
        raise ValueError("Vector database is empty — call build_vector_db() first.")
    
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    
    # Pre-compute query norm for efficiency
    query_norm = sum(x*x for x in query_embedding) ** 0.5
    
    similarities = []
    for chunk, embedding in VECTOR_DB:
        # Compute embedding norm once per chunk
        embedding_norm = sum(x*x for x in embedding) ** 0.5
        
        if query_norm == 0 or embedding_norm == 0:
            score = 0.0
        else:
            dot_product = sum(x*y for x, y in zip(query_embedding, embedding))
            score = dot_product / (query_norm * embedding_norm)
        
        similarities.append((chunk, score))
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]

def create_prompt(retrieved_knowledge, query):
    """Create a grounded prompt with retrieved context."""
    if not retrieved_knowledge:
        context = "No relevant information found in the knowledge base."
    else:
        context = "\n".join(f"- {chunk}" for chunk, _ in retrieved_knowledge)
    
    instruction_prompt = f"""You are a grounded question-answering assistant.
Use only the context below to answer the user's question.
If the context does not contain enough evidence, say that the answer is not in the knowledge base.
When records conflict, prefer a clearly dated newer record and explain the update briefly.

Context:
{context}
"""
    return instruction_prompt

def answer_question(query, top_n=3):
    """Full RAG pipeline: retrieve, construct prompt, and generate answer."""
    # Retrieve relevant chunks
    retrieved_knowledge = retrieve(query, top_n=top_n)
    
    # Display retrieved knowledge
    print("\nRetrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        print(f"- ({similarity:.3f}) {chunk}")
    
    # Create grounded prompt
    instruction_prompt = create_prompt(retrieved_knowledge, query)
    
    # Generate answer and capture it
    full_answer = ""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )
    
    print("\nAnswer:")
    for response_chunk in stream:
        token = response_chunk["message"]["content"]
        print(token, end="", flush=True)
        full_answer += token
    print()
    
    return full_answer, retrieved_knowledge

def build_vector_db(filename):
    """Build the vector database from a file."""
    global VECTOR_DB
    VECTOR_DB = []
    
    print(f"Loading {filename}...")
    dataset = load_dataset(filename)
    print(f"Loaded {len(dataset)} chunks")
    if len(dataset) >= 2:
        print(f"First two chunks: {dataset[0][:50]}... and {dataset[1][:50]}...")
    elif len(dataset) == 1:
        print(f"First chunk: {dataset[0][:50]}...")
    
    # Checkpoint 1: Confirm blank lines were not included
    print(f"✅ Checkpoint 1: Blank lines excluded (dataset length = {len(dataset)})")
    
    print("\nBuilding vector database...")
    for chunk in dataset:
        add_chunk_to_database(chunk)
    
    # Checkpoint 2: Verify VECTOR_DB matches dataset
    print(f"✅ Checkpoint 2: len(VECTOR_DB) = {len(VECTOR_DB)} == len(dataset) = {len(dataset)}")
    if VECTOR_DB:
        print(f"✅ Checkpoint 2: First embedding is a list of {len(VECTOR_DB[0][1])} numbers")
    
    print(f"Vector database contains {len(VECTOR_DB)} chunks with embeddings")

def run_evaluation(filename, questions):
    """Run evaluation with a specific knowledge base."""
    # Check file exists before starting evaluation
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Skipping evaluation for this file.")
        return []
    
    build_vector_db(filename)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"Question {i}: {question}")
        print(f"{'='*50}")
        answer, retrieved = answer_question(question, top_n=3)
        results.append({
            "question": question,
            "answer": answer,
            "retrieved": retrieved
        })
    
    return results

def manual_testing():
    """Run manual testing with several queries before evaluation."""
    print("\n=== Manual Testing Mode ===")
    print("Testing retrieval with sample queries...")
    
    test_queries = [
        "cat anatomy",
        "sleep patterns",
        "cat behavior"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            retrieved = retrieve(query, top_n=2)
            print("Retrieved chunks:")
            for chunk, similarity in retrieved:
                print(f"- ({similarity:.3f}) {chunk[:80]}...")
        except ValueError as e:
            print(f"Error: {e}")
        print()

def main():
    """Main function to run the RAG system."""
    # Test cosine similarity
    test_cosine_similarity()
    
    # Manual testing mode
    manual_testing()
    
    # Define evaluation questions
    cat_facts_questions = [
        "What additional organ allows a cat to smell besides its nose, and where is that organ located?",
        "A cat is nine years old. Based on the knowledge base, approximately how many years of its life has it been awake?"
    ]
    
    confusing_questions = [
        "Which Luna needs rabbit-based food, and what color collar does she wear?",
        "Can Juniper attend the adoption event scheduled for 2026-05-02? Explain the evidence for your answer."
    ]
    
    # Collect results for README table
    all_results = []
    
    # Run evaluation with cat-facts.txt
    print("=== EVALUATION WITH cat-facts.txt ===")
    cat_facts_results = run_evaluation("cat-facts.txt", cat_facts_questions)
    all_results.extend([("cat-facts.txt", i+1, 3, r["answer"], r) for i, r in enumerate(cat_facts_results)])
    
    # Run evaluation with cat-db-fiction-confusing.txt
    print("\n\n=== EVALUATION WITH cat-db-fiction-confusing.txt ===")
    confusing_results = run_evaluation("cat-db-fiction-confusing.txt", confusing_questions)
    all_results.extend([("cat-db-fiction-confusing.txt", i+1, 3, r["answer"], r) for i, r in enumerate(confusing_results)])
    
    # Print summary for README table
    print("\n\n=== SUMMARY FOR README TABLE ===")
    for file, q_num, top_n, answer, _ in all_results:
        print(f"File: {file}, Question: {q_num}, Answer: {answer[:100]}...")

if __name__ == "__main__":
    main()
