# Summarizing-using-LaMini-Flan-T5-248M

This project utilizes the LaMini Flan T5 model, fine-tuned on Google/flan-t5-base, for text summarization tasks. The goal is to provide a user-friendly interface for summarizing text inputs.

## Steps

1. **Choosing a Suitable Model**:
   The first challenge was to select a model suitable for local usage with a moderate parameter count. LaMini Flan T5 model was chosen for its fine-tuning on Google/flan-t5-base, striking a balance between performance and resource requirements.

2. **Word Embeddings from FAAIS**:
   To enhance the model's understanding of the text, FAAIS  was utilized to select word embeddings, enriching the vocabulary and semantic representation.

3. **Creating a Model Pipeline with Langhain**:
   Langhain was instrumental in constructing a robust pipeline for the model. Leveraging its capabilities, we seamlessly integrated preprocessing, model execution, and post-processing steps to ensure efficient summarization.

4. **Deployment with Streamlit**:
   The summarization tool was deployed using Streamlit, offering a user-friendly interface. Streamlit's simplicity and versatility allowed us to create an intuitive application, enabling users to input text and obtain concise summaries effortlessly.


