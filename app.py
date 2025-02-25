import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize LLM with a free Hugging Face model
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.5, "max_length": 500})

# Define a summarization prompt
template = "Summarize the following text: {text}"
prompt = PromptTemplate(template=template, input_variables=["text"])
summarization_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("AI-Powered Research Assistant")
st.subheader("Summarize Research Papers & Answer Queries")

# Text Input
user_input = st.text_area("Enter text (or paste a research paper snippet):", "")

if st.button("Summarize"):
    if user_input:
        summary = summarization_chain.run(text=user_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

# Define a QA prompt
template_qa = "Based on the following context, answer the question: {question} Context: {context}"
prompt_qa = PromptTemplate(template=template_qa, input_variables=["question", "context"])
qa_chain = LLMChain(llm=llm, prompt=prompt_qa)

# QA Section
st.subheader("Ask a Question About the Research")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question and user_input:
        answer = qa_chain.run(question=question, context=user_input)
        st.write("**Answer:**", answer)
    else:
        st.warning("Please enter both text and a question.")

# Footer
st.markdown("---")
st.markdown("Powered by Streamlit & LangChain with Hugging Face Models")
