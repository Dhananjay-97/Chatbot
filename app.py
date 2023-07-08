import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import os

pinecone.init(api_key=os.environ['02c657ba-59f6-488b-847f-7912cc1b544c'], environment=os.environ['asia-southeast1-gcp-free'])
index_name = "fieldmanual"

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['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'])
pinecone_instance = Pinecone.from_existing_index(index_name, embeddings)

openAI = OpenAI(temperature=0, openai_api_key=os.environ['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'])

chain = load_qa_chain(openAI, chain_type="stuff")


def askGPT(prompt):
    docs = pinecone_instance.similarity_search(prompt, include_metadata=True)
    ch = chain.run(input_documents=docs, question=prompt)
    print(ch)


def main():
    while True:
        print('Open AI + Pinecone: Field Manual Querying\n')
        prompt = "prompt:" + input()
        
        askGPT(prompt)
        print('\n')


if __name__ == "__main__":
    main()
