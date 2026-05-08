from langchain_core.language_models.chat_models import BaseChatModel
from abc import ABC, abstractmethod


class BaseRAG(ABC):
    """
    Interface base para implementações de sistemas RAG.
    """

    def __init__(self, llm_instance: BaseChatModel, **kwargs):
        """        
        Args:
            llm_instance: Instância pré-carregada de um modelo de linguagem BaseChatModel (ex.: ChatOpenAI, ChatHuggingFace, etc.)
            **kwargs: Parâmetros adicionais (temperature, max_new_tokens, etc.).
        """
        assert isinstance(llm_instance, BaseChatModel), "llm_instance deve ser uma instância de BaseChatModel"
        self.llm_instance = llm_instance
        self.params = kwargs

    # função auxiliar
    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gera uma resposta para a pergunta fornecida usando o modelo de linguagem.

        Args:
            system_prompt (str): O prompt do sistema.
            user_prompt (str): O prompt do usuário.

        Returns:
            str: A resposta gerada pelo modelo de linguagem.
        """
        if system_prompt is None or system_prompt.strip() == "":
            chat_history = [
                ("user", user_prompt)
            ]
        else:
            chat_history = [
                ("system", system_prompt),
                ("user", user_prompt)
            ]
        response = self.llm_instance.invoke(chat_history, **self.params)
        return response.text

    @abstractmethod
    def answer_question(self, question: str) -> str:
        """
        Processa uma pergunta e retorna a resposta gerada pelo RAG.

        Args:
            question (str): A pergunta a ser respondida.

        Returns:
            str: A resposta gerada pelo RAG.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Libera os recursos de hardware utilizados especificamente pelo RAG.
        
        Deve limpar a base de vetores e referências a documentos da memória,
        garantindo que o hardware esteja limpo para a próxima execução.
        """
        pass

