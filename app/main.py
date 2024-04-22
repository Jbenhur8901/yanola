from fastapi import FastAPI, Form, UploadFile
from app.agent import RetrievalAgent
from app.basicLLM import YanolaBasic, YanolaKeyFacts
from app.document_processing import url_loaders, process_file, delete
import uvicorn

app = FastAPI()


@app.get("/")
async def health_check():
    """
    Perform a health check.

    This endpoint is used to check the health status of the application.
    
    Returns:
        dict: A JSON response indicating the status of the application. 
              Example: {"status": "ok"}

    NB : Intellectual Property of Nodes Technology
    """
    return {"status": "ok"}

@app.post("/basic_agent")
async def yanola_basic(query: str = Form(...),
                       session_id: str = Form(...)):
    
    """
    Perform inference using the Yanola Basic Model.

    This endpoint is used to perform inference using the Yanola Basic Model,
    which is a basic language model for handling text input.

    Args:
        query (str): The text query to be processed by the model.
        session_id (str): A unique identifier for the session.

    Returns:
        dict: A JSON response containing the model's response to the query.
              Example: {"response": "Model response goes here"}

    NB : Intellectual Property of Nodes Technology
    """
    agent = YanolaBasic(user_input=query,session_id=session_id)
    response = agent.run()
    return {"response": response}

@app.post("/expert_agent")
async def expert_agent(query: str = Form(...),
                       index_name: str = Form(...),
                       sys_instruction: str = Form(...),
                       chat_id: str = Form(...),
                       subject_matter: str = Form(...),
                       model: str = Form("gpt-3.5-turbo-0125"),
                       timeline: int = Form(300),
                       voyageai_model: str = Form("voyage-large-2"),
                       temperature: float = Form(0.5)
                       ):
    """
    Perform inference using the RetrievalAgent.

    This endpoint is used to perform inference using the RetrievalAgent,
    which handles more complex queries and interactions.

    Args:
        query (str): The text query to be processed by the agent.
        index_name (str): The name of the index to be searched.
        sys_instruction (str): System instruction for the agent.
        chat_id (str): Identifier for the chat session.
        subject_matter (str): The subject matter of the query.
        model (str): The name of the language model to use (default: "gpt-3.5-turbo-0125").
        timeline (int): The timeline for the agent's response (default: 300).
        voyageai_model (str): The name of the VoyageAI model to use (default: "voyage-large-2").
        temperature (float): The temperature parameter for text generation (default: 0.5).

    Returns:
        dict: A JSON response containing the agent's response to the query.
              Example: {"response": "Agent response goes here"}

    NB : Intellectual Property of Nodes Technology
    """
    agent = RetrievalAgent(user_input = query,
                           index_name = index_name,
                           sys_instruction = sys_instruction,
                           chat_id = chat_id,
                           subject_matter = subject_matter,
                           model=model,
                           timeline = timeline,
                           voyageai_model = voyageai_model,
                           temperature = temperature)
    response = agent.run_agent()
    return {"response": response}

@app.post("/process_url")
async def process_url(url: str = Form(...),
                       index: str = Form(...)):
    """
    Process a URL and extract subject matter.

    This endpoint is used to process a URL and extract relevant subject matter.

    Args:
        url (str): The URL of the document to be processed.
        index (str): The index name for storing the extracted subject matter.

    Returns:
        dict: A JSON response containing the extracted subject matter.
              Example: {"response": "Subject matter goes here"}

    NB : Intellectual Property of Nodes Technology
    """
    
    subjet_matter = url_loaders(urls = url,
                                index=index)
    
    return {"response": subjet_matter}

@app.post("/process_file")
async def process_file(file: UploadFile):
    """
    Process a file and extract information.

    This endpoint is used to process a file and extract relevant information,
    such as index IDs.

    Args:
        file (UploadFile): The file to be processed.

    Returns:
        dict: A JSON response containing information about the processed file.
              Example: {"index_ids": [id1, id2, ...], "filename": "filename", "size": size}

    NB : Intellectual Property of Nodes Technology
    """
    ids = process_file(file.file)

    return {"index_ids":ids,"filename":file.filename,"size":file.size}

@app.post("/keyfacts_agent")
async def yanolakey_facts(query: str = Form(...),
                       session_id: str = Form(...)):
    
    """
    Perform inference using the Yanola Key Facts Extraction Model.

    This endpoint is used to perform inference using the Yanola Key Facts Extraction Model,
    which is specialized in extracting key facts and essential information from text input.

    Args:
        query (str): The text query to be processed by the model.
        session_id (str): A unique identifier for the session.

    Returns:
        dict: A JSON response containing the model's response to the query.
              Example: {"response": "Model response goes here"}
              
    NB: Intellectual Property of Nodes Technology
    """
    agent = YanolaKeyFacts(user_input=query,session_id=session_id)
    response = agent.run()
    return {"response": response}

@app.post("/delete")
async def process_file(index_ids: list = Form(...),
                       index_name: str = Form(...)):
    """
    Delete documents from an index.

    This endpoint is used to delete documents from an index based on their IDs.

    Args:
        index_ids (list): A list of document IDs to be deleted.
        index_name (str): The name of the index from which documents should be deleted.

    Returns:
        dict: A JSON response indicating the status of the deletion operation.
              Example: {"response": "Deletion status goes here"}
    
    NB : Intellectual Property of Nodes Technology
    """
    status = delete(ids = index_ids,
                 index_name = index_name)

    return {"response":status}



# if __name__ == "__main__":
#     uvicorn.run(app,host="localhost", port=8000,reload=True)