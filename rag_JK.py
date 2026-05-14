"""
Implementação do RAG sobre a biografia de Juscelino Kubitschek.

Este arquivo contém a classe JK_RAG, que herda de BaseRAG.
A Knowledge Base já deve ter sido criada previamente pelo script build_kb.py
e salva na pasta vector_db/.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
        **kwargs,
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
            allow_dangerous_deserialization=True,
        )

        print("RAG carregado com sucesso!")

    def _buscar_chunks_relevantes(self, question: str):
        """
        Busca os chunks mais relevantes usando MMR.

        O MMR tenta equilibrar relevância e diversidade,
        evitando recuperar vários chunks muito parecidos entre si.

        Args:
            question (str): Pergunta feita pelo usuário.

        Returns:
            list: Lista de documentos/chunks recuperados.
        """
        retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": max(30, self.k * 5),
                "lambda_mult": 0.5,
            },
        )
        
        documentos = retriever.invoke(question)

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
        Chama o modelo de linguagem para gerar a resposta.

        Esta versão chama diretamente o modelo recebido em llm_instance,
        evitando problemas de diferença de nome entre métodos auxiliares
        como _generate_response ou self_generate_response.
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

        if hasattr(response, "content"):
            return response.content

        if hasattr(response, "text"):
            return response.text

        return str(response)

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
Use exclusivamente as informações presentes no CONTEXTO RECUPERADO.
Não use conhecimento externo.
Não complete lacunas com suposições.
Não invente datas, cargos, instituições, nomes, partidos ou eventos.
Não mencione fatos que não estejam explicitamente no contexto.

Se o contexto recuperado não trouxer a resposta de forma clara, responda exatamente:
"Não encontrei informação suficiente no contexto recuperado."

Quando responder, cite a página ou as páginas usadas como base, se elas estiverem disponíveis no contexto.
A resposta deve ser objetiva e fiel ao texto.
"""

        user_prompt = f"""
CONTEXTO RECUPERADO:
{contexto}

PERGUNTA:
{question}

INSTRUÇÕES PARA A RESPOSTA:
- Responda apenas com base no CONTEXTO RECUPERADO.
- Se a informação não aparecer claramente no contexto, diga:
  "Não encontrei informação suficiente no contexto recuperado."
- Não use conhecimento externo.
- Não invente informações.
- Não generalize além do que está escrito no contexto.
- Quando possível, mencione a página usada.

RESPOSTA:
"""

        resposta = self._gerar_resposta(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt.strip(),
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