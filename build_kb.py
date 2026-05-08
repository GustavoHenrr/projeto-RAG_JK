"""
Script para criar a Knowledge Base (KB) do projeto RAG.

Este script:
1. Carrega o PDF completo da biografia de Juscelino Kubitschek;
2. Divide o texto em chunks;
3. Gera embeddings para cada chunk;
4. Cria uma base vetorial usando FAISS;
5. Salva a base vetorial na pasta vector_db/.

Observação:
Este script deve ser executado apenas quando você quiser criar ou recriar a KB.
O RAG em si não deve recriar a KB toda vez que for responder perguntas.
"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Configurações de caminhos
# ============================================================

# Caminho da raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent

# Caminho do PDF da biografia
PDF_PATH = PROJECT_ROOT / "data" / "biografia_juscelino.pdf"

# Pasta onde a base vetorial será salva
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"


# ============================================================
# Configurações da Knowledge Base
# ============================================================

# Modelo de embeddings multilíngue.
# Como o documento e as perguntas estão em português, usamos um modelo adequado para várias línguas.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Tamanho dos chunks.
# Esse valor pode ser testado depois nos experimentos.
CHUNK_SIZE = 1000

# Sobreposição entre chunks.
# Ajuda a não perder contexto entre um trecho e outro.
CHUNK_OVERLAP = 150


def carregar_pdf(pdf_path: Path):
    """
    Carrega o PDF completo usando PyMuPDFLoader.

    Args:
        pdf_path (Path): Caminho do arquivo PDF.

    Returns:
        list: Lista de documentos carregados, geralmente um por página.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF não encontrado em: {pdf_path}\n"
            "Verifique se o arquivo biografia_juscelino.pdf está dentro da pasta data/."
        )

    print(f"Carregando PDF: {pdf_path}")

    loader = PyMuPDFLoader(str(pdf_path))
    documentos = loader.load()

    print(f"Total de páginas/documentos carregados: {len(documentos)}")

    return documentos


def preparar_metadados(documentos):
    """
    Ajusta os metadados dos documentos.

    O PyMuPDFLoader normalmente guarda a página no metadado 'page',
    começando em 0. Aqui criamos também um campo 'page_number',
    começando em 1, para facilitar citar páginas nas respostas.

    Args:
        documentos (list): Lista de documentos carregados.

    Returns:
        list: Lista de documentos com metadados ajustados.
    """

    for doc in documentos:
        pagina_zero_based = doc.metadata.get("page")

        if pagina_zero_based is not None:
            doc.metadata["page_number"] = pagina_zero_based + 1

        doc.metadata["source"] = "biografia_juscelino.pdf"

    return documentos


def dividir_em_chunks(documentos):
    """
    Divide os documentos em chunks menores.

    Args:
        documentos (list): Lista de documentos carregados do PDF.

    Returns:
        list: Lista de chunks.
    """

    print("Dividindo documentos em chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
         add_start_index=True
    )

    chunks = splitter.split_documents(documentos)

    print(f"Total de chunks criados: {len(chunks)}")

    return chunks


def criar_e_salvar_vector_db(chunks):
    """
    Cria embeddings, monta a base vetorial FAISS e salva em disco.

    Args:
        chunks (list): Lista de chunks gerados a partir do PDF.
    """

    print("Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("Criando base vetorial FAISS...")
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print(f"Salvando base vetorial em: {VECTOR_DB_PATH}")
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    vector_db.save_local(str(VECTOR_DB_PATH))

    print("Knowledge Base criada e salva com sucesso!")


def main():
    """
    Função principal do script.
    """

    documentos = carregar_pdf(PDF_PATH)

    documentos = preparar_metadados(documentos)

    chunks = dividir_em_chunks(documentos)

    criar_e_salvar_vector_db(chunks)


if __name__ == "__main__":
    main()