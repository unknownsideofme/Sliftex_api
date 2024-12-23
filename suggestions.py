from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
import json

def reformat_suggestions(output):
    try:
        # Check if the input is a string
        if isinstance(output, str):
            # Remove any triple backticks
            output = output.strip('`')
            # Parse the input JSON string
            parsed_output = json.loads(output)
        else:
            # If input is not a string, assume it's already a dict
            parsed_output = output

        # Check if the key is wrapped in an unnecessary string
        if isinstance(parsed_output, dict) and "suggestions" in parsed_output:
            suggestions = parsed_output["suggestions"]

            # Ensure keys are formatted correctly
            reformatted_suggestions = {
                key.strip().rstrip(".") + ".": value.strip()
                for key, value in suggestions.items()
            }

            return {"suggestions": reformatted_suggestions}

    except json.JSONDecodeError:
        # If JSON parsing fails, return the original output
        return output

    return output  # Return untouched if format doesn't matc




def calc_suggestions(res, title , llm) :
    # Convert JSON string into a Document object
    if isinstance(res, str):
        res = json.loads(res)

    # Convert `res` into LangChain-compatible Document objects
    res_document = [
        Document(page_content=json.dumps(value), metadata={"type": key})
        for key, value in res.items()
    ]
    docs = []
    for doc in res_document:
    # Split the page_content of the document
        chunks = text_splitter.split_text(doc.page_content)
    # Recreate Document objects for each chunk, keeping the original metadata
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=doc.metadata))
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a title verification assistant for the Press Registrar General of India. 
        1. You will be given a response that stores the list pf phonetically  similar and semantically similar exisitng titles corresponding  to the input title.
        2. Your task is to give suggestions to the user based on the phonetic similarity, semantic similarity and prefix and suffix  so that the title achieves a better acceptance score.
        3. Suggestions can include removing or replacing commonly used prefixes and suffixes or making the title more phonetically and semantically unique.
        4. An example suggestion can be like "The word Shakti in your given title has been repeated many times" if the word Shakti had been repeated many times or is semantically similar to many existing titles.
        5. An example suggetsion can be like "Try removing the prefixes or suffixes or replacing them" if prefix suffix had been repeated many times
        
        6. Do not generate any greeting or assisting or any other message except for the suggestions in pointers. 
        7. Please return the data in the json format provided below where the suggestions are in the form of a json **object** not string with the key as the suggestion number and the value as the suggestion.
        8. Never repeat same type of suggestion again and again and give atleast 5 suggestions.
        
        Output Format:
        {{
            "suggestions": {{
                
                "1." : "Suggestion 1",
                "2." : "Suggestion 2",
                ....
            }}
        }}
        
        
        Input: {input}
        Context: {context}
        """
    )
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Invoke the document chain
    response = document_chain.invoke({"input": title, "context": docs})
    response = reformat_suggestions(response)
    
    # Parse the response and return suggestions
    return response
