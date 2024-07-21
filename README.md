# Evaluating the Ideal Document Chunk Size for a RAG Application with Qdrant

This project evaluates the ideal document chunk size for a Retrieval-Augmented Generation (RAG) application using Qdrant. It explores how chunk size impacts performance and demonstrates advanced techniques to balance precision and context retention.

## Features

- **Interactive UI**: A Streamlit-based web interface that is easy to use.
- **Chunk Size Evaluation**: Evaluate the impact of different chunk sizes on query precision, context relevancy, and retrieval performance.
- **Advanced Retrieval Techniques**: Implements Context Enrichment with Sentence Window Retrieval for improved performance.
- **Vector Storage**: Utilizes Qdrant to store and retrieve vector embeddings, ensuring efficient and scalable search capabilities.
- **Real-Time Results**: Delivers immediate search results, offering users a dynamic interaction with the application.

## How to Use

### Set Up Your Environment

1. **Ensure Python and Streamlit are installed.**
2. **Clone the repository and install dependencies.**

    ```sh
    git clone https://github.com/your-username/rag-chunk-size-evaluation.git
    cd rag-chunk-size-evaluation
    pip install -r requirements.txt
    ```

### Configure Qdrant

1. **Create a Qdrant Cluster**: Follow the steps outlined in the Qdrant documentation to create a cluster in the Qdrant Cloud. [Qdrant Cloud Quickstart](https://qdrant.tech/documentation/cloud/quickstart-cloud/)
2. **Set Up Environment Variables**: Create a `.env` file in the root directory and add your OpenAI API key, Qdrant URL, and Qdrant API key.

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    QDRANT_API_KEY=your_qdrant_api_key
    QDRANT_URL=your_qdrant_url
    ```

### Prepare the Qdrant Collection

Determine the name for your Qdrant collection where the document embeddings will be stored. If you're running the application for the first time, ensure to run the code that creates and populates the collection.

### Launch the Application

Execute the following command to start the app:

```sh
streamlit run app.py
```

### Explore Results
Upload your documents and evaluate the performance with different chunk sizes through the Streamlit UI.

### Installation
1. Clone the repository:
```
git clone https://github.com/AI-ANK/Evaluating-the-Ideal-Document-Chunk-Size-for-a-RAG-Application.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a .env file in the root directory of the project and add your OpenAI API key, Qdrant URL, and Qdrant API key. Replace the placeholder values with your actual keys:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
```

## Tools and Technologies
- **UI**: Streamlit
- **Vector Store**: Qdrant
- **LLM Orchestration**: Llamaindex
- **LLM**: GPT 3.5 Turbo

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please open an issue in the GitHub issue tracker.

## Authors
### Developed by [Harshad Suryawanshi](https://www.linkedin.com/in/harshadsuryawanshi/)
If you find this project useful, consider giving it a ⭐ on GitHub!
