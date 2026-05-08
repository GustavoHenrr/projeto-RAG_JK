from base_rag import BaseRAG

##################################################################
##  EXEMPLO DE COMO IMPLEMENTAR UM RAG HERDANDO DA CLASSE BASE  ##
##################################################################
'''
Atenção, a classe abaixo faria só o fluxo padrão de RAG, sem o processo
inicial de indexação (dividir em chunks, criar vector deb, etc).

O processo inicial de indexação deve ser feito à parte (em um scrip separado),
uma única vez, salvando o vector db em arquivo, que deve ser uploaded para o
seu repositório.
'''

class MyRAG(BaseRAG):
    def __init__(self, llm_instance, param1, param2, **kwargs):
        # carregar o vector db (que deve ter sido criado em arquivo anteriormente)
        # carregar modelos de re-rank, embedding, ...
        super().__init__(llm_instance, **kwargs)

    def answer_question(self, question: str) -> str:
        # Buscar os chunks mais relevantes...

        # Outras etapas (re-rank, etc...)

        # Montar um prompt
        myprompt = "..."

        # Enviar para o modelo e receber a resposta
        # => pode chamar diretamente o modelo assim: 
        # resposta = self.llm_instance.invoke([("user", question)])
        # => ou pode usar essa função auxiliar
        resposta = self._generate_response(system_prompt='', prompt=myprompt)
        
        return resposta

    def teardown(self) -> None:
        # Desalocar recursos como o vector db, etc...
        print("Recursos de hardware liberados.")



# Exemplo de como instanciar um modelo do HuggingFace e instanciar o RAG

if __name__ == "__main__":
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    MODEL_ID = "google/gemma-2b-it"

    local_llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="text-generation",
        pipeline_kwargs=dict(
            do_sample=True,
            max_new_tokens=2048,
            return_full_text=False  # Atenção: Precisa setar este valor para contornar um bug!!!
        )
    )
    chat_model = ChatHuggingFace(llm=local_llm)

    rag = MyRAG(llm_instance=chat_model)

    resposta = rag.answer_question("Juscelino foi a Recife alguma vez?")
    print(resposta)
