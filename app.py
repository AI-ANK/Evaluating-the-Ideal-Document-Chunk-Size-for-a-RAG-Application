import streamlit as st
import os
import pandas as pd
import nest_asyncio
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from qdrant_client import QdrantClient
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    context_entity_recall
)
from ragas.metrics.critique import harmfulness
from ragas.integrations.llama_index import evaluate
from datasets import Dataset

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

nest_asyncio.apply()

# Load environment variables from a .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')

# Streamlit UI
st.title("RAG Pipeline with RAGAS Evaluation and Advanced Retrieval Techniques")

# Upload documents
# uploaded_files = st.file_uploader("Upload Documents", type=["txt"], accept_multiple_files=True)
if st.button("Start Analysis"):
    with st.spinner("Loading documents..."):
        documents = SimpleDirectoryReader("./document").load_data()

    # Generator with OpenAI models
    generator_llm = OpenAI(model="gpt-3.5-turbo")
    critic_llm = OpenAI(model="gpt-3.5-turbo")
    embeddings = OpenAIEmbedding()

    df_loaded = pd.read_pickle("testset.pkl")
    st.write("Loaded DataFrame columns:", df_loaded.columns)
    

    # Build query engine
    qdrant_client = QdrantClient(location=":memory:")
    # st.write(qdrant_client.get_collections())

    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="chunk_analyzer")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex.from_documents(documents)
    query_engine = vector_index.as_query_engine()

    ds = Dataset.from_pandas(df_loaded)

    ds_dict = ds.to_dict()
    # st.write("Dataset Questions:", ds_dict["question"])
    # st.write("Ground Truth Answers:", ds_dict["ground_truth"])
        # Display Dataset Questions and Ground Truth Answers
    st.subheader("Dataset Questions and Ground Truth Answers")
    df_questions_answers = pd.DataFrame({
        "Question": ds_dict["question"],
        "Ground Truth Answer": ds_dict["ground_truth"]
    })
    st.table(df_questions_answers)
    # st.stop()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
        context_entity_recall,
    ]

    evaluator_llm = OpenAI(model="gpt-3.5-turbo-0125")

    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=ds_dict,
        llm=evaluator_llm,
        embeddings=OpenAIEmbedding(),
        raise_exceptions=False
    )

    rdf = result.to_pandas()
    st.write("Evaluation Results:", rdf.head())

    chunk_sizes = [25, 1024, 2000]
    results = []

    for chunk_size in chunk_sizes:
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = int(0.05 * chunk_size)

        index = VectorStoreIndex.from_documents(documents)
        query_engine2 = index.as_query_engine(similarity_top_k=2)

        result = evaluate(
            query_engine=query_engine2,
            metrics=metrics,
            dataset=ds_dict,
            llm=evaluator_llm,
            embeddings=OpenAIEmbedding(),
        )

        rdf = result.to_pandas()
        rdf['chunk_size'] = chunk_size
        results.append(rdf)

        st.write(f"Results for chunk size {chunk_size}:", rdf)

    all_results_df = pd.concat(results, ignore_index=True)
    # st.write("Combined results:")
    # st.write(all_results_df[['context_recall', 'context_precision', 'context_relevancy', 'context_entity_recall']])
    
    # Display combined results with chunk size
    st.subheader("Combined Results")
    st.write(all_results_df[['chunk_size', 'context_recall', 'context_precision', 'context_relevancy', 'context_entity_recall']])


    documents = SimpleDirectoryReader("./document").load_data()

    # Initialize Qdrant client
    client = QdrantClient(
        location=":memory:"

    )

    # Create two separate vector stores for sentence window and base indices
    sentence_vector_store = QdrantVectorStore(client=client, collection_name="sentence_window_collection")
    base_vector_store = QdrantVectorStore(client=client, collection_name="base_collection")

    # Create storage contexts
    sentence_storage_context = StorageContext.from_defaults(vector_store=sentence_vector_store)
    base_storage_context = StorageContext.from_defaults(vector_store=base_vector_store)

    # Initialize text splitter and LLM
    text_splitter = SentenceSplitter()
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Update Settings
    Settings.llm = llm
    Settings.text_splitter = text_splitter
    Settings.chunk_size = 50
    Settings.chunk_overlap = 10

    # Create node parser
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Get nodes from documents
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes = text_splitter.get_nodes_from_documents(documents)

    # Create indices using Qdrant vector stores
    sentence_index = VectorStoreIndex(nodes, storage_context=sentence_storage_context)
    base_index = VectorStoreIndex(base_nodes, storage_context=base_storage_context)

    # Create query engine with sentence window retrieval
    query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
    )

    # window_response = query_engine.query(
    #     "What are the potential applications and challenges for future research related to technitium?"
    # )
    # st.write("Window Response:", window_response)

    # window = window_response.source_nodes[0].node.metadata["window"]
    # sentence = window_response.source_nodes[0].node.metadata["original_text"]

    # st.write(f"Window: {window}")
    # st.write("------------------")
    # st.write(f"Original Sentence: {sentence}")

    # query_engine = base_index.as_query_engine(similarity_top_k=2)
    # vector_response = query_engine.query(
    #     "What are the potential applications and challenges for future research related to technitium"
    # )
    # st.write("Vector Response:", vector_response)

    user_question = "What are the potential applications and challenges for future research related to technitium?"
    user_question = df_questions_answers.iloc[0]["Question"]

    window_response = query_engine.query(user_question)

    window = window_response.source_nodes[0].node.metadata["window"]
    sentence = window_response.source_nodes[0].node.metadata["original_text"]

    # Displaying window and vector responses clearly
    st.subheader("Query Results")
    st.write(f"**User Question:** {user_question}")
    st.write("**Response using Sentence Window Retrieval:**")
    st.write(window_response.response)
    st.divider()
    st.markdown(f"**Window:** \n\n {window}")
    st.markdown(f"**Original Sentence:** \n\n {sentence}")

    query_engine = base_index.as_query_engine(similarity_top_k=2)
    vector_response = query_engine.query(user_question)
    st.divider()
    st.write("**Response using Normal Retrieval:**")
    st.write(vector_response.response)

    st.write("**Source Nodes used for Normal Retrieval:**")
    for source in vector_response.source_nodes:
        st.write(source)

