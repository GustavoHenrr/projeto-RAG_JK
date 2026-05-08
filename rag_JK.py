"""
Implementação do RAG sobre a biografia de Juscelino Kubitschek.

Este arquivo contém a classe JK_RAG, que herda de BaseRAG.
A Knowledge Base já deve ter sido criada previamente pelo script build_kb.py
e salva na pasta vector_db/.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS # Usado para carregar a base vetorial criada no build_kb.py
from langchain_huggingface import HuggingFaceEmbeddings # Usado para transformar as perguntas em vetores

from base_rag import BaseRAG


class JK_RAG(BaseRAG):
    """
    RAG para responder perguntas sobre a biografia de Juscelino Kubitschek.

    Esta classe:
    1. Carrega a base vetorial FAISS salva em disco;
    2. Recupera os chunks mais relevantes para uma pergunta;
    3. Monta um prompt com o contexto recuperado;
    4. Usa o modelo de linguagem recebido no construtor para gerar a resposta.
    """

    def __init__(
        self,
        llm_instance,
        vector_db_path: str = "vector_db",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        k: int = 5,
        **kwargs
    ):
        """
        Inicializa o RAG.

        Args:
            llm_instance: Modelo de linguagem compatível com LangChain.
            vector_db_path (str): Caminho da pasta onde a base FAISS foi salva.
            embedding_model_name (str): Nome do modelo de embeddings usado.
            k (int): Quantidade de chunks recuperados para cada pergunta.
            **kwargs: Parâmetros adicionais enviados ao modelo de linguagem.
        """

        super().__init__(llm_instance, **kwargs)

        self.k = k
        self.embedding_model_name = embedding_model_name

        project_root = Path(__file__).resolve().parent
        self.vector_db_path = project_root / vector_db_path

        if not self.vector_db_path.exists():
            raise FileNotFoundError(
                f"Base vetorial não encontrada em: {self.vector_db_path}\n"
                "Execute primeiro o script build_kb.py para criar a Knowledge Base."
            )

        print("Carregando modelo de embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        print(f"Carregando base vetorial em: {self.vector_db_path}")
        self.vector_db = FAISS.load_local(
            folder_path=str(self.vector_db_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("RAG carregado com sucesso!")

    def _buscar_chunks_relevantes(self, question: str):
        """
        Busca os chunks mais relevantes na base vetorial.

        Args:
            question (str): Pergunta feita pelo usuário.

        Returns:
            list: Lista de documentos/chunks recuperados.
        """

        documentos = self.vector_db.similarity_search(
            query=question,
            k=self.k
        )

        return documentos

    def _formatar_contexto(self, documentos):
        """
        Formata os chunks recuperados em um único texto de contexto.

        Args:
            documentos (list): Lista de chunks recuperados.

        Returns:
            str: Contexto formatado para ser enviado ao LLM.
        """

        partes_contexto = []

        for i, doc in enumerate(documentos, start=1):
            pagina = doc.metadata.get("page_number", "página não informada")
            conteudo = doc.page_content.strip()

            trecho = (
                f"[Trecho {i} | Página {pagina}]\n"
                f"{conteudo}"
            )

            partes_contexto.append(trecho)

        contexto = "\n\n".join(partes_contexto)

        return contexto

    def _gerar_resposta(self, system_prompt: str, user_prompt: str) -> str:
        """
        Chama o modelo de linguagem usando a função auxiliar da BaseRAG.

        Observação:
        No PDF e no example_rag.py aparece o nome _generate_response.
        Porém, no base_rag.py fornecido, o método está como self_generate_response.
        Por isso usamos self_generate_response aqui.
        """

        return self.self_generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    def answer_question(self, question: str) -> str:
        """
        Responde uma pergunta usando o fluxo RAG.

        Args:
            question (str): Pergunta sobre a biografia de Juscelino Kubitschek.

        Returns:
            str: Resposta gerada pelo modelo com base no contexto recuperado.
        """

        if question is None or question.strip() == "":
            return "A pergunta está vazia. Por favor, envie uma pergunta válida."

        documentos = self._buscar_chunks_relevantes(question)

        if len(documentos) == 0:
            return "Não encontrei informação suficiente no contexto recuperado."

        contexto = self._formatar_contexto(documentos)

        system_prompt = """
Você é um assistente acadêmico especializado em responder perguntas sobre a biografia de Juscelino Kubitschek.

Responda em português.
Use apenas as informações presentes no contexto fornecido.
Não use conhecimento externo.

Se o contexto não trouxer informação suficiente para responder com segurança, diga claramente:
"Não encontrei informação suficiente no contexto recuperado."

Sempre que possível, mencione as páginas do PDF usadas como base.
"""

        user_prompt = f"""
Contexto recuperado:

{contexto}

Pergunta:
{question}

Resposta:
"""

        resposta = self._gerar_resposta(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt.strip()
        )

        return resposta

    def teardown(self) -> None:
        """
        Libera referências usadas pelo RAG.

        Isso não apaga a base vetorial do disco.
        Apenas remove da memória os objetos carregados nesta instância.
        """

        self.vector_db = None
        self.embeddings = None

        print("Recursos do RAG liberados.")