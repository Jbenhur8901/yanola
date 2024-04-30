import logging
from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException, status
from app.agent import RetrievalAgent
from app.basicLLM import YanolaBasic, YanolaKeyFacts
from app.document_processing import url_loaders, delete
import uvicorn
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import requests
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"])

base_id = os.environ["AIRTABLE_BASE_ID"]
table_id = os.environ["AIRTABLE_TABLE_ID"]
url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
auth_token = f"{os.environ['AIRTABLE_API_KEY']}"

headers = {"Authorization": f"Bearer {auth_token}"}

def airtable_get_data():
    """
    Fetches data from Airtable using the Airtable API.

    This function sends a GET request to the specified Airtable API URL with the provided headers,
    retrieves the response data, and returns it as a JSON object.

    Returns:
        dict: A dictionary containing the response data from Airtable, parsed as JSON.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while fetching data from Airtable: {e}")
        return []  

def json_db():
    """
    Constructs a JSON-like database from data retrieved from Airtable.

    Retrieves data from Airtable using the `airtable_get_data()` function,
    then constructs a dictionary-like database where each record's username
    serves as the key, and the record's fields, including an added 'id' field
    with the record's ID, serve as the corresponding values.

    Returns:
        dict: A dictionary representing the constructed database, with usernames
              as keys and corresponding record fields as values, including the record ID.
    """
    try:
        data = airtable_get_data()
    except Exception as e:
        logger.error(f"Error fetching data from Airtable: {e}")

    db = {}
    for record in data["records"]:
        record_username = record["fields"]["username"]
        record_fields = record["fields"]
        record_id = record["id"]
        record_fields["id"] = record_id
        db[record_username] = record_fields
    return db
    
db = json_db()

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str = None

class User(BaseModel):
    username: str
    email: str = None
    full_name: str = None
    disabled: bool = None

class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token:str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Initialize the FastAPI app
app = FastAPI()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

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
                       session_id: str = Form(...),
                       current_user: User = Depends(get_current_active_user)):
    
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
    if db[current_user.username]["limit"] > 0 :
        db[current_user.username]["limit"]-=1
        agent = YanolaBasic(user_input=query,session_id=session_id)
        response = agent.run()
        return {"response": response,"limit":db[current_user.username]["limit"]}
    else:
        return {"response": "Vous avez atteint votre limite mensuelle"}

@app.post("/expert_agent")
async def expert_agent(query: str = Form(...),
                       index_name: str = Form(...),
                       sys_instruction: str = Form(...),
                       chat_id: str = Form(...),
                       subject_matter: str = Form(...),
                       model: str = Form("gpt-3.5-turbo-0125"),
                       timeline: int = Form(300),
                       voyageai_model: str = Form("voyage-large-2"),
                       temperature: float = Form(0.5),
                       current_user: User = Depends(get_current_active_user)
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
    if db[current_user.username]["limit"] > 0 :
        db[current_user.username]["limit"]-=1
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
    
    else:
        return {"response": "Vous avez atteint votre limite mensuelle"}


@app.post("/process_url")
async def process_url(url: str = Form(...),
                       index: str = Form(...),
                       current_user: User = Depends(get_current_active_user)):
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
async def process_file(file: UploadFile = File(...),
                       current_user: User = Depends(get_current_active_user)):
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
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_pinecone import PineconeVectorStore
    from langchain_voyageai import VoyageAIEmbeddings
    import os

    filename = file.filename
    path = "tempfiles/{filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    embedding = VoyageAIEmbeddings(model="voyage-large-2")
    loader = PyPDFLoader(path)
    docs = loader.load()

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators="\n\n")
        chunks = text_splitter.split_documents(docs)
    except Exception as e:
        logger.error(f"Error when processing documents: {e}")
    
    try:
        vector = PineconeVectorStore(index_name="urls", embedding=embedding)
        ids = vector.add_documents(chunks)
        status = "Success"
    except Exception as e:
        logger.error(f"Error adding documents to index: {e}")
    
    try:
        # Remove the file
        os.remove(path)
    except OSError as e:
        logger.error(f"Error removing the file {e}")

    return {"filename":filename, "status":status, 'length': len(ids)}

@app.post("/keyfacts_agent")
async def yanolakey_facts(query: str = Form(...),
                       session_id: str = Form(...),
                       current_user: User = Depends(get_current_active_user)):
    
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
    if db[current_user.username]["limit"] > 0 :
        db[current_user.username]["limit"]-=1  
        agent = YanolaKeyFacts(user_input=query,session_id=session_id)
        response = agent.run()
        return {"response": response}
    else:
        return {"response": "Vous avez atteint votre limite mensuelle"}
    
@app.post("/delete")
async def delete(index_ids: list = Form(...),
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

if __name__ == "__main__":
   uvicorn.run(app)