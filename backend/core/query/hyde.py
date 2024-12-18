import numpy as np
import time
import openai
from sentence_transformers import SentenceTransformer


# Define prompt templates for different tasks
LEGAL_ADVICE = """Vui lòng cung cấp ý kiến pháp lý cho trường hợp sau theo luật pháp Việt Nam, chỉ cung cấp một ý khoảng 50 token.
Trường hợp: {}
Ý kiến pháp lý:"""""

LEGAL_SUMMARY = """Vui lòng tóm tắt văn bản pháp lý sau đây.
Văn bản: {}
Tóm tắt:"""""

LEGAL_QUESTION = """Vui lòng cung cấp câu trả lời cho câu hỏi pháp lý sau theo luật pháp Việt Nam.
Câu hỏi: {}
Câu trả lời:"""""

# Promptor class to create prompts for specific tasks
class Promptor:
    def __init__(self, task: str, language: str = "en"):
        """
        Initializes the Promptor for a specific task and language.
        Args:
            task (str): The task type (e.g., 'legal advice', 'legal summary', 'legal question').
            language (str): Language of the prompt (default: 'en').
        """
        self.task = task
        self.language = language

    def build_prompt(self, query: str) -> str:
        """
        Builds a prompt based on the task type and query.
        Args:
            query (str): User's input or question.
        Returns:
            str: The formatted prompt for the task.
        """
        if self.task == "legal advice":
            return LEGAL_ADVICE.format(query)
        elif self.task == "legal summary":
            return LEGAL_SUMMARY.format(query)
        elif self.task == "legal question":
            return LEGAL_QUESTION.format(query)
        else:
            raise ValueError("Task not supported")


# Base Generator class for interacting with different APIs
class Generator:
    def __init__(self, model_name: str, api_key: str):
        """
        Initializes the generator with a model name and API key.
        Args:
            model_name (str): Name of the AI model.
            api_key (str): API key for authentication.
        """
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        """
        Placeholder method for generating text. Subclasses should implement their own generation logic.
        Args:
            prompt (str): The input prompt.
        Returns:
            str: A default response indicating the method is not yet implemented.
        """
        return "This is a default response from the Generator class. Please implement the generate method in a subclass."


# OpenAIGenerator class to interact with OpenAI's API
import openai
import time

class OpenAIGenerator(Generator):
    def __init__(self, model_name: str, api_key: str, n: int = 1, max_tokens: int = 100, temperature: float = 0.7,
                 top_p: float = 1, frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 stop: list = None, wait_till_success: bool = False):
        """
        Initializes the OpenAIGenerator with parameters for text generation.
        """
        super().__init__(model_name, api_key)
        openai.api_key = api_key
        self.model_name = model_name
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop or ["\n\n\n"]
        self.wait_till_success = wait_till_success

    @staticmethod
    def parse_response(response: dict) -> list:
        """
        Parses the response from OpenAI and ranks the generated texts.
        Args:
            response (dict): The raw response from OpenAI.
        Returns:
            list: Generated texts.
        """
        return [choice["message"]["content"] for choice in response.get("choices", [])]

    def generate(self, prompt: str) -> list:
        """
        Sends the prompt to OpenAI and retrieves the generated response.
        Args:
            prompt (str): The input prompt.
        Returns:
            list: A list of generated texts.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,  # Sử dụng "gpt-3.5-turbo" làm model_name nếu bạn muốn
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    n=self.n,
                )
                return self.parse_response(response)
            except openai.error.OpenAIError as openai_error:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise openai_error
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred: {str(e)}")


# HyDE class to handle the full process
class HyDE:
    def __init__(self, promptor: Promptor, generator: Generator, encoder: SentenceTransformer, searcher):
        """
        Initializes HyDE with promptor, generator, encoder, and searcher.
        """
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher

    def prompt(self, query: str) -> str:
        """
        Builds a prompt for the input query.
        """
        return self.promptor.build_prompt(query)

    def generate(self, query: str) -> list:
        """
        Generates hypothesis documents based on the query.
        """
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents

    def encode(self, query: str, hypothesis_documents: list) -> dict:
        """
        Encodes the query and hypothesis documents into vectors and returns as a dictionary.
        """
        # Encode the query
        query_embedding = self.encoder.encode(query)

        # Encode hypothesis documents
        hypothesis_embeddings = [self.encoder.encode(doc) for doc in hypothesis_documents]

        # Create a dictionary to hold the embeddings
        embeddings_dict = {
            "query": query_embedding
        }

        # Add each hypothesis document to the dictionary with a unique key
        for idx, embedding in enumerate(hypothesis_embeddings):
            embeddings_dict[f"hypothesis_{idx}"] = embedding

        # Optionally, calculate the average embedding (you can add it to the dictionary if needed)
        avg_embedding = np.mean([query_embedding] + hypothesis_embeddings, axis=0)
        embeddings_dict["average_embedding"] = avg_embedding

        return embeddings_dict

    def search(self, hyde_vector: np.ndarray, k: int = 10):
        """
        Performs search based on the encoded vector.
        """
        return self.searcher.search(hyde_vector, k=k)

    def e2e_search(self, query: str, k: int = 10):
        """
        Performs the end-to-end process: generate -> encode -> search.
        """
        hypothesis_documents = self.generate(query)
        hyde_vector = self.encode(query, hypothesis_documents)
        return self.search(hyde_vector, k=k)
