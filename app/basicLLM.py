from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

class YanolaBasic:
    def __init__(self, user_input, session_id) -> None:

        self.llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
        self.url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        self.user_input = user_input
        self.session_id = session_id
        self.prompt = ChatPromptTemplate.from_messages([
        ("system","""Tu es Yanola, une assistante virtuelle créée par Nodes Technology, une startup 
        de la Republique du Congo spécialisée dans l'Intélligence artificielle. Tu es là pour aider
        du mieux que tu peux avec toutes sortes de tâches et de questions, du moment que cela reste
        dans les limite de ce qui est légal et éthique."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","<input>{input}</input>")
        ])
        

    def run(self) -> str :
        upstash_client = UpstashRedisChatMessageHistory(url=self.url, token=self.token, ttl=300, session_id=self.session_id)
        memory = ConversationBufferMemory(chat_memory=upstash_client,memory_key="chat_history", return_messages=True,)
        chain = LLMChain(llm = self.llm, prompt= self.prompt, memory=memory) 
        output = chain.invoke({"input":self.user_input})
        return output["text"]
    

class YanolaKeyFacts:
    def __init__(self, user_input, session_id) -> None:

        self.llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
        self.url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        self.user_input = user_input
        self.session_id = session_id
        self.prompt = ChatPromptTemplate.from_messages([
        ("system","""Tu es Yanola, une experte dans l'extraction d'informations pertinentes et l'exploration de textes. Tu possèdes
        un esprit analytique aigu, capable de disséquer des données textuelles complexes et d'en extraire des informations significatives de manière efficace.
        Tu accorde une attention méticuleuse aux détails, garantissant l'exactitude et la précision du processus d'extraction afin d'obtenir des résultats de haute qualité.
        Ton objectif est d'améliorer les processus de prise de décision et accroître la productivité des organisations.  """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","<input>{input}</input>")
        ])
        

    def run(self) -> str :
        upstash_client = UpstashRedisChatMessageHistory(url=self.url, token=self.token, ttl=300, session_id=self.session_id)
        memory = ConversationBufferMemory(chat_memory=upstash_client,memory_key="chat_history", return_messages=True,)
        chain = LLMChain(llm = self.llm, prompt= self.prompt, memory=memory) 
        output = chain.invoke({"input":self.user_input})
        return output["text"]


