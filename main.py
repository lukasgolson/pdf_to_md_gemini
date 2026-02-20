import streamlit as st
import PyPDF2
import io
import google.generativeai as genai

st.set_page_config(page_title="PDF to Markdown Converter", page_icon="📄", layout="wide")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key:", type="password")
    st.caption("*Get your free key at [Google AI Studio](https://aistudio.google.com/).*")

    st.divider()

    st.subheader("Advanced Settings")
    model_name = st.selectbox(
        "Model Version",
        ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.1-pro-preview"],
        index=0,
        help="Flash-Lite is best for cost/speed. Pro is best for complex formatting."
    )

    st.divider()

    if st.button("🗑️ Clear App Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")


# --- CACHED FUNCTIONS ---

# We pass file_bytes instead of the file object because Streamlit caches bytes more reliably
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes):
    """Reads the PDF file bytes and extracts raw text."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


@st.cache_data(show_spinner=False)
def format_text_to_markdown(raw_text, api_key, selected_model):
    """Sends raw text to the LLM and caches the result to prevent duplicate billing."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(selected_model)

        prompt = """
        You are an expert document formatter. I am going to provide you with raw text extracted from a PDF article. 
        Your job is to read the entire text, strip away weird artifacts (like page numbers, headers, footers, and broken line breaks), 
        and output a clean, highly readable, well-structured Markdown version of the article.

        Add appropriate markdown headings (##, ###), bullet points, and bold text where it makes logical sense.
        Do not add any conversational filler or introductions. Output ONLY the raw Markdown text.

        Here is the PDF text:
        ====================
        """

        response = model.generate_content(prompt + raw_text)

        in_tokens = response.usage_metadata.prompt_token_count
        out_tokens = response.usage_metadata.candidates_token_count

        return response.text, in_tokens, out_tokens

    except Exception as e:
        st.error(f"Error during LLM formatting: {e}")
        return None, 0, 0


# --- MAIN UI ---
st.title("📄 PDF to E-Reader Markdown")
st.write("Upload a PDF article. The AI will clean up the formatting and output a Markdown file.")

uploaded_file = st.file_uploader("Upload your PDF article", type=["pdf"])

if uploaded_file is not None:
    if not api_key:
        st.warning("👈 Please enter your API key in the sidebar to proceed.")
    else:
        if st.button("Convert to Markdown", type="primary"):

            with st.status("Processing Document...", expanded=True) as status:
                st.write("1️⃣ Extracting raw text from PDF...")
                # Read file as bytes for reliable caching
                file_bytes = uploaded_file.getvalue()
                raw_text = extract_text_from_pdf(file_bytes)

                if raw_text:
                    st.write(f"2️⃣ Sending to {model_name} for formatting...")
                    markdown_result, input_tokens, output_tokens = format_text_to_markdown(raw_text, api_key,
                                                                                           model_name)

                    if markdown_result:
                        status.update(label="Conversion Complete!", state="complete", expanded=False)

                        # Calculate Costs ($0.10/1M input and $0.40/1M output for Flash-Lite)
                        # Note: Cost math would need to change dynamically if you switch to Pro models
                        input_cost = (input_tokens / 1_000_000) * 0.10
                        output_cost = (output_tokens / 1_000_000) * 0.40
                        total_cost = input_cost + output_cost

                        # Store in session state for UI rendering
                        st.session_state['markdown'] = markdown_result
                        st.session_state['stats'] = (input_tokens, output_tokens, total_cost)
                    else:
                        status.update(label="Failed at LLM stage.", state="error")
                else:
                    status.update(label="Failed to extract text.", state="error")

# --- RESULTS UI ---
if 'markdown' in st.session_state:
    in_tok, out_tok, cost = st.session_state['stats']

    st.subheader("📊 Conversion Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Input Tokens", f"{in_tok:,}")
    col2.metric("Output Tokens", f"{out_tok:,}")
    col3.metric("Estimated Cost (Flash-Lite)", f"${cost:.5f}")

    st.divider()

    st.subheader("Preview")
    with st.expander("Click to view Markdown preview", expanded=True):
        st.markdown(st.session_state['markdown'])

    st.divider()

    filename = uploaded_file.name.replace('.pdf', '.md')
    st.download_button(
        label="⬇️ Download Markdown File",
        data=st.session_state['markdown'],
        file_name=filename,
        mime="text/markdown",
        type="primary",
        use_container_width=True
    )