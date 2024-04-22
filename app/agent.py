
#Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
#Prompts
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
#Agents
from langchain.agents import AgentExecutor, create_tool_calling_agent
#Vectordb
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
#Memory
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
# Document Compression
from langchain.retrievers.document_compressors import LLMChainExtractor
# Prompt
from langchain.prompts import PromptTemplate
#Tools
from langchain_community.tools.tavily_search import TavilySearchResults
#Custom tools
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
#Query Transformation 
from langchain.retrievers.multi_query import MultiQueryRetriever
#Environnement Variable
import logging
from dotenv import load_dotenv
import os

load_dotenv()

class RetrievalAgent:
    """
    Retrieval agent for extracting relevant documents based on user queries.
    
    This class manages the retrieval process, including fetching chat history, setting up retrieval tools, and executing the agent.
    """

    def __init__(self, user_input: str, index_name: str, sys_instruction: str, chat_id: str, subject_matter: str, model="gpt-3.5-turbo-0125",timeline: int = 300, voyageai_model: str = "voyage-large-2", temperature: float = 0.5):
        """
        Initialize the RetrievalAgent.

        Parameters:
        - user_input (str): The user query.
        - index_name (str): The name of the vector index.
        - sys_instruction (str): System instruction message.
        - chat_id (str): Unique identifier for the chat session.
        - subject_matter (str): Subject matter of the queries.
        - timeline (int): Time limit for chat history retrieval.
        - voyageai_model (str): Name of the VoyageAI model.
        - temperature (float): Temperature parameter for language model.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:
            self.model = model
            if self.model.startswith("gpt"):
                self.llm = ChatOpenAI(model_name=self.model,temperature=temperature)
            elif self.model.startswith("claude"):
                self.llm = ChatAnthropic(model_name=model,temperature=temperature)
            else:
                self.llm = ChatOpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ["TOGETHER_API_KEY"], model=model)
            self.embedding = VoyageAIEmbeddings(model=voyageai_model)
            self.user_input = user_input
            self.index_name = index_name
            self.sys_instruction = sys_instruction
            self.subject_matter = subject_matter
            self.chat_id = chat_id
            self.timeline = timeline
            self.get_chat_history()
            self.set_retrieval_tool()
            self.set_tools()
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")

    def get_chat_history(self):
        """
        Retrieve the chat history from the storage.
        """
        try:
            url = os.getenv("UPSTASH_REDIS_REST_URL")
            token = os.getenv("UPSTASH_REDIS_REST_TOKEN")+"="
            self.upstash_client = UpstashRedisChatMessageHistory(url=url, token=token, ttl=self.timeline, session_id=self.chat_id)
            self.chat_history = self.upstash_client.messages
        except Exception as e:
            self.logger.error(f"Error fetching chat history: {e}")
            self.chat_history = []

    def nodes_pipeline(self, question) -> list:
        """
        Pipeline for extracting relevant documents based on the user's question.

        This method processes the user's question to generate alternative versions and extract relevant documents from a vector database.
        It utilizes language models and retrieval techniques to enhance document retrieval based on user queries.

        Parameters:
        - question (str): The user's question.

        Returns:
        - List of relevant documents.
        """
        try:
            # Define prompts for query transformation and context compression
            prompt_query = """Tu es un assistant de modèle linguistique d'IA. Ta tâche consiste à générer quatre versions différentes de la question posée par l'utilisateur afin d'extraire les documents pertinents 
            d'une base de données vectorielle. En générant des perspectives multiples sur la question de l'utilisateur,ton objectif est d'aider l'utilisateur à surmonter certaines des limites de la recherche de similarité
            basée sur la distance. Propose ces questions alternatives en les séparant par des nouvelles lignes.
        
            question originale: {question}
            """
            prompt_compressor = """Considérant la question et le contexte suivants, extraire toute partie du contexte *EN L'ÉTAT* qui est pertinente pour répondre à la question. Si aucune partie du contexte n'est pertinente,
            renvoie NO_OUTPUT. \n\n - Tu ne dois jamais modifier les parties extraites du contexte.\n- Question : {question}\n- Contexte:\n>>{context}\n>>\nParties pertinentes extraites :"""
            query_transformation_prompt = PromptTemplate(template=prompt_query, input_variables=["question"])
            context_compression_prompt = PromptTemplate(template=prompt_compressor, input_variables=["question", "context"])

            # Set up vector store and retriever
            vector_store = PineconeVectorStore(embedding=self.embedding, index_name=self.index_name)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Initialize language model for query transformation
            llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0)

            # Transform query and compress context
            retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, prompt=query_transformation_prompt)
            unique_docs = retriever_from_llm.get_relevant_documents(question)
            compressor = LLMChainExtractor.from_llm(llm=llm, prompt=context_compression_prompt)
            self.compressed_docs = compressor.compress_documents(documents=unique_docs, query=question)
            return self.compressed_docs
        except Exception as e:
            # Log error and return empty list
            self.logger.error(f"Error in nodes_pipeline: {e}")
        return []

    
    def set_retrieval_tool(self):
        """
        Set up the retrieval tool.
        """
        try:
            class PipelineInputs(BaseModel):
                question : str = Field(description=f"La question posée sur {self.subject_matter}")

            self.nodes_pipe = StructuredTool.from_function(func=self.nodes_pipeline,
                                                  name="nodes_pipeline",
                                                  description="""Cet outil utilise des techniques de recherche avancées pour récupérer les documents pertinents 
                                                  à partir d'une base de données vectorielles en fonction des questions posées. Il est indispensable d'utiliser 
                                                  cet outil pour obtenir le contexte pertinent lorsque l'on répond à des questions rélatives à {self.subject_matter}.""",
                                                  args_schema=PipelineInputs
                                                  )
        except Exception as e:
            self.logger.error(f"Error setting up retrieval tool: {e}")

    def set_tools(self):
        """
        Set up additional tools.
        """
        try:
            self.tools = [self.nodes_pipe]
        except Exception as e:
            self.logger.error(f"Error setting up additional tools: {e}")

    def run_agent(self):
        """
        Run the retrieval agent.
        """
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                ("system", f"{self.sys_instruction}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "<input>{input}</input>"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
                ]
            )

            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            query = executor.invoke({"input": self.user_input, "chat_history": self.chat_history})
            self.upstash_client.add_messages([HumanMessage(content=self.user_input), AIMessage(content=query["output"])])
        except Exception as e:
            self.logger.error(f"Error running agent: {e}")

        return query["output"]


