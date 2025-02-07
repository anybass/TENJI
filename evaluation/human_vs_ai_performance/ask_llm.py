#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:21:39 2025

"""

from py2neo import Graph
import openai
import tiktoken
import time

# Enter your OpenAI API key
openai.api_key = "ENTER_YOUR_API_KEY"
# Connect to the Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", "ENTER_YOUR_PASSWORD"))

# Function to load prompt from a text file
def load_prompt(filename):
    """Loads a prompt from a text file."""
    try:
        with open(f"prompts/{filename}.txt", "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename}.txt not found in prompts folder.")
        return ""
    
# Define prompt templates
qa_prompt_template = load_prompt("qa_prompt")
generate_cypher_prompt_template = load_prompt("generate_cypher")
regenerate_alternative_cypher_prompt_template = load_prompt("retry_generating_cypher")
summarization_prompt_template = load_prompt("summarization")

# Define the OpenAI GPT-4 API call using the new method
def openai_gpt4_api_call(prompt, temperature=0, max_tokens=700):
    """Direct call to OpenAI GPT-4 using the chat model endpoint"""
    response = openai.chat.completions.create(
        model="gpt-4",  # GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# Query Neo4j function
def query_graph(cypher_query):
    """Executes a Cypher query and returns the results."""
    return graph.run(cypher_query).data()

# Function to query LLM for Cypher query generation
def generate_cypher_query(question):
    # Provide a prompt to the LLM to generate a Cypher query
    prompt = generate_cypher_prompt_template.format(question=question)
  
    # Call OpenAI API to generate the Cypher query
    response = openai.chat.completions.create(
        model="gpt-4",  # You can use different models depending on your setup
        messages=[
            {"role": "system", "content": "You are a German data scientist."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    query = response.choices[0].message.content.strip("```").replace("cypher","").strip()  # Extract the Cypher query from the response
    return query

# Token counting function
def count_tokens(prompt, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))    


def incremental_summary(chunks, question,  model="gpt-4"):
    summary = ""
    print("Chunks to be processed: ")
    print(len(chunks))
    for i, chunk in enumerate(chunks):
        # Add document identification to the prompt for each chunk
        document_info = f"Document {i+1}"  # You can replace this with your actual document ID logic if needed
        prompt = summarization_prompt_template(question=question, summary=summary, document_info=document_info, chunk=chunk)
        response = openai_gpt4_api_call(prompt)
        summary = response.strip()  # Update summary with LLM response
        print(f"Summary after processing chunk {i+1}:\n{summary}\n")
    
    return summary

def extract_document_identifier(result):
    """
    Extract the document identifier (such as case number) from the result.
    This assumes that the result is a dictionary and the identifier is stored in a specific field like 'number'.
    Modify the extraction logic based on your result format.
    """
    # Assuming the result is a dictionary with a key 'number' or similar for the document identifier
    return result.get("c.number", "Unknown Document ID")

def split_into_chunks_with_identifiers(results, model="gpt-4", token_limit=7000):
    """
    Split the context into chunks that respect the model's token limit while maintaining document boundaries.
    Only split documents that exceed the token limit, preserving each document's context.
    """
    chunks = []
    
    for result in results:
        # Extract document identifier and text
        document_id = extract_document_identifier(result)
        document_text = str(result)  # Assuming this is the text of the document
        
        # Check the token count of the document
        context_tokens = count_tokens(f"Document ID: {document_id}\n{document_text}", model)
        
        # If the document exceeds the token limit, split it into chunks
        if context_tokens > token_limit:
            print(f"Document {document_id} exceeds token limit, splitting into smaller chunks...")
            # Now split into chunks while maintaining the document boundary
            tokens_per_chunk = token_limit // 2  # Leave room for model output tokens
            start = 0
            while start < len(document_text):
                end = min(start + tokens_per_chunk, len(document_text))
                chunk = f"Document ID: {document_id}\n{document_text[start:end]}"
                chunks.append(chunk)
                start = end
        else:
            # If the document is within the limit, just add it as a whole
            chunks.append(f"Document ID: {document_id}\n{document_text}")
    
    return chunks

def answer_question_with_retries(
    question, 
    use_knowledge_graph=True, 
    max_retries=3,
    token_limit=7000,  # Set a token limit close to the model's input limit
    model="gpt-4"
):
    """Answers a question, retrying with alternative Cypher queries if no results are found."""
    queries=[]
    if use_knowledge_graph:
        retry_count = 0
        results = []
        error_message = None  # To capture errors
        
        while retry_count < max_retries:
            # Step 1: Generate the Cypher query
            if retry_count == 0:
                cypher_query = generate_cypher_query(question)
            else:
                # Retry with alternative Cypher query, including error feedback
                alt_prompt = regenerate_alternative_cypher_prompt_template.format(question=question, cypher_query=cypher_query, error_message=error_message)
                cypher_query = openai_gpt4_api_call(alt_prompt).strip("```").replace("cypher", "").strip()
            
            print(f"Attempt {retry_count + 1}: Generated Cypher Query:\n{cypher_query}")
            
            # Step 2: Execute the query and check results
            try:
                queries.append(cypher_query)
                query_result = query_graph(cypher_query)
                if "error" in query_result:
                    error_message = query_result["error"]  # Capture the error for the next retry
                    print(f"Query error: {error_message}")
                else:
                    results = query_result
                    if results:
                        print("Results found:")
                        print(results)
                        break  # Exit the loop if results are found
            except Exception as e:
                # Catching errors related to the execution of the Cypher query
                error_message = str(e)
                print(f"Execution failed with error: {error_message}")
            
            # Retry if query fails or no results
            if not results:
                print("No results found or query failed. Retrying with a new query...")
            retry_count += 1
        
        if not results:
            return {"response":"Nach mehreren Versuchen konnten keine validen Ergebnisse ermittelt werden.", "queries":queries}
                

         # Step 3: Construct the context from the query results with document identifiers
        # Each result already includes the document ID in the format "Document ID: <identifier>\n<result>"
        context_with_ids = "\n".join([f"Document ID: {extract_document_identifier(result)}\n{str(result)}" for result in results])

        
        # Step 4: Check token count and summarize if necessary
        if count_tokens(context_with_ids, model) > token_limit:
            print("Context exceeds token limit. Summarizing in chunks...")
            # Split the results into smaller chunks and summarize incrementally while maintaining document identifiers
            chunks = split_into_chunks_with_identifiers(results, model, token_limit)
            context = incremental_summary(chunks, question, model)
        else:
            # If within token limit, no need to chunk, use the full context
            context = context_with_ids

        # Use the summarized or full context to construct the prompt
        prompt = qa_prompt_template.format(context=context, question=question, query=cypher_query)
    else:
        # If not using the knowledge graph, ask the LLM directly
        prompt = f"Frage: {question}\n\nAntwort:"

    # Step 5: Use the chosen LLM model to generate the answer
    response = openai_gpt4_api_call(prompt)

    return {"response":response, "queries":queries}





if __name__ == "__main__":
    
    #short text tasks 
    questions=["Geben Sie den Wortlaut von Art. 33 V GG an.", "Nennen Sie drei Entscheidungen, in denen Art. 33 GG eine wesentliche Rolle spielt.", "Nennen sie den Leitsatz des Beschlusses „Geschlechtsumwandlung“ aus dem Jahr 1993 (BVerfGE 88, 87).","Nennen sie zwei Grundgesetz Artikel, die auf Art. 1 GG verweisen.","Nennen Sie die Fundstelle (BVerfGE) der Entscheidung mit dem Namen „Nassauskiesung“.","War die Verfassungsbeschwerde im Fall „Reiten im Walde“ erfolgreich?","Nennen Sie drei Entscheidungen, auf die die Entscheidung „Schächterlaubnis“ referiert.","Nennen Sie drei Entscheidungen, die auf die Entscheidung „Schächterlaubnis“ referieren.","Das BVerfG hat folgende Formulierung entwickelt: „Differenzierungen bedürfen stets der Rechtfertigung durch Sachgründe, die dem Differenzierungsziel und dem Ausmaß der Ungleichbehandlung angemessen sind“ - Nennen sie drei Entscheidungen, in der diese Formulierung vorkommt.","Im Urteil Nichtraucherschutzgesetz (BVerfGE 121, 317) schreibt das Gericht: „Die aus Gründen des Gemeinwohls unumgänglichen Einschränkungen der Berufsfreiheit stehen unter dem Gebot der Verhältnismäßigkeit“ – diese Aussage wird mit drei anderen Entscheidungen belegt: Nennen sie diese.", "Wie viele Abschnitte sind in „Abschnitt 1 Allgemeine Grundrechtslehren“ des Buchs „Grundrechte-Klausur-und Examenswissen“ enthalten?", "Nennen Sie drei Grundgesetz Artikel, mit denen das Wort „Deutschengrundrechte“ in dem Buch „Grundrechte-Klausur-und Examenswissen“ in Verbindung gebracht wird?", "Welche Referenzen zu Normen oder Urteilen werden in der Fallfrage zu Fall 1 im Buch „Grundrechte-Klausur-und Examensfälle“ gemacht?", "Welchen Artikel zitiert Art. 19 GG?", "Mit welchen Grundgesetz Artikeln wird Art. 14 GG in den beiden Büchern „GrundrechteKlausur-und Examenswissen“ und „Grundrechte-Klausur-und Examensfälle“ häufig gemeinsam referenziert?", "Welcher Grundgesetz Artikel regelt „Diskriminierungsverbot aufgrund der Religion bei staatsbürgerlichen Rechten“?", "In welchem Jahr wurde das Elfes-Urteil gefällt?", "Welcher Grundgesetz Artikel beinhaltet „Die Bestimmungen der Artikel 136, 137, 138, 139 und 141 der deutschen Verfassung vom 11. August 1919 sind Bestandteil dieses Grundgesetzes.“?", "Welcher Paragraph des BayVfGHG wird in „§ 16 Landesverfassungen“ im Buch „Grundrechte-Klausur-und Examenswissen“ referenziert?", "Nennen Sie einen Grundgesetz Artikel, der vom Apotheken-Urteil zitiert wird."] 
    
    #exam
    #questions=["Sachverhalt A war erfolgreiche Profi-Leichtathletin. Sie ist in der ehemaligen DDR aufgewachsen, und hat als Kind dort bereits Leistungssport getrieben. Bei den Olympischen Spielen 1992 in Barcelona hat sie eine Silbermedaille gewonnen. 2022 berichtet die Tageszeitung Z, die durch die eingetragene Z-Gesellschaft bürgerlichen Rechts (Z-eGbR) verlegt wird, von systematischem Doping schon bei Kindern in der DDR. Die Z schreibt: „A hat als 14jährige das verbotene Doping-Mittel O verabreicht bekommen. Dies wirft einen dunklen Schatten auch auf ihre späteren Erfolge.“ A klagt gegen die Z-eGbR auf Zahlung eines Schmerzensgeldes von 10.000 EUR wegen der Berichterstattung. Im Prozess vor den Zivilgerichten sagt die ehemalige Trainingspartnerin P als Zeugin aus. Sie gibt zu Protokoll, zwar nicht gesehen zu haben, wie A gedopt habe. Sie sei sich aber sicher, dass nicht nur sie selbst, sondern die gesamte Trainingsgruppe das verbotene Dopingmittel habe einnehmen müssen. Die Zivilgerichte geben der Schmerzensgeldklage der A statt, weil nicht mit letzter Gewissheit zivilprozessual bewiesen sei, dass auch A das Mittel O eingenommen habe. Die Z-eGbR erhebt daraufhin Verfassungsbeschwerde. Ist die zulässige Verfassungsbeschwerde begründet?   Auszug aus dem Bürgerlichen Gesetzbuch (BGB): § 823 Schadensersatzpflicht (1) Wer vorsätzlich oder fahrlässig das Leben, den Körper, die Gesundheit, die Freiheit, das Eigentum oder ein sonstiges Recht eines anderen widerrechtlich verletzt, ist dem anderen zum Ersatz des daraus entstehenden Schadens verpflichtet. (2) [...] § 253 Immaterieller Schaden (1) Wegen eines Schadens, der nicht Vermögensschaden ist, kann Entschädigung in Geld nur in den durch das Gesetz bestimmten Fällen gefordert werden. (2) Ist wegen einer Verletzung des Körpers, der Gesundheit, der Freiheit oder der sexuellen Selbstbestimmung Schadensersatz zu leisten, kann auch wegen des Schadens, der nicht Vermögensschaden ist, eine billige Entschädigung in Geld gefordert werden. § 705 Rechtsnatur der Gesellschaft (1) [...] (2) Die Gesellschaft kann entweder selbst Rechte erwerben und Verbindlichkeiten eingehen, wenn sie nach dem gemeinsamen Willen der Gesellschafter am Rechtsverkehr teilnehmen soll (rechtsfähige Gesellschaft), oder sie kann den Gesellschaftern zur Ausgestaltung ihres Rechtsverhältnisses untereinander dienen (nicht rechtsfähige Gesellschaft). (3) Ist der Gegenstand der Gesellschaft der Betrieb eines Unternehmens unter gemeinschaftlichem Namen, so wird vermutet, dass die Gesellschaft nach dem gemeinsamen Willen der Gesellschafter am Rechtsverkehr teilnimmt."]
    
    for q, question in enumerate(questions):
         print(question)
         # Start time for the question
         start_time = time.time()
         use_knowledge_graph = True # Set to False to bypass the knowledge graph
         data = answer_question_with_retries(question, use_knowledge_graph=use_knowledge_graph)
         answer=data["response"]
         cypher_queries=data["queries"]
         # End time for the question
         end_time = time.time()

         # Calculate execution time
         execution_time = end_time - start_time
         print(f"Execution time for question {q+1}: {execution_time:.4f} seconds")
         print("Answer:", answer)
         with open("./llm_response/answers_openai_gpt4_kg.txt", "a", encoding="utf8") as file:
              file.write(str(str(q)+": "+str(answer)+"\n\n"))
         with open("./llm_response/cypherqueries_openai_gpt4_kg.txt", "a", encoding="utf8") as file2:
              file2.write(str(str(q)+": "+str(cypher_queries)+"\n\n"))
         with open("./llm_response/time_openai_gpt4_kg.txt", "a", encoding="utf8") as file3:
             file3.write(str(str(q)+": "+str(execution_time)+"\n\n"))
         print()
