import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="PDF to Markdown Converter", page_icon="📄", layout="wide")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key:", type="password")

    st.divider()
    model_name = st.selectbox(
        "Model Version",
        ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.1-pro-preview"],
        index=1,  # Defaulting to 1.5-flash for better complex table/math handling
        help="Pro is best for complex formatting and math. Flash is a good middle ground."
    )

    run_validation = st.checkbox(
        "Run Validation Pass (Slower, 2x Cost)",
        value=False,
        help="Makes a second LLM call to cross-check the generated Markdown against the original PDF for transcription errors."
    )

    st.divider()
    if st.button("🗑️ Clear App Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")


# --- CACHED FUNCTIONS ---

@st.cache_data(show_spinner=False)
def format_pdf_to_markdown(file_bytes, mime_type, api_key, selected_model):
    """Sends raw PDF bytes directly to Gemini to clean and format into Markdown."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(selected_model)

        # We pass the raw document to Gemini so it retains spatial/visual context
        document_part = {
            "mime_type": mime_type,
            "data": file_bytes
        }

        prompt = """
        You are an expert document formatter and meticulous data transcriber. I am providing you with a raw PDF document. 
        Your job is to read the entire document and output a clean, highly readable, well-structured Markdown version.

        CRITICAL INSTRUCTIONS:
        1. 100% DATA FIDELITY: You must transcribe all numbers, coordinates, dates, and measurements EXACTLY as they appear. Do not round, abstract, or alter any numerical data.
        2. MATHEMATICS & SYMBOLS: Pay extreme attention to minus signs (-), operators, and complex formulas. Do not drop negative signs. For formal/complex math or science equations, use LaTeX enclosed in $ for inline and $$ for display equations. Do not use LaTeX for simple numbers.
        3. TABLES: Transcribe tables with absolute perfection. Ensure columns align perfectly and no data is shifted into the wrong row/column. 
        4. FORMATTING: Strip away page numbers, headers, footers, and broken line breaks. Add logical Markdown headings (##, ###).

        Do not add any conversational filler or introductions. Output ONLY the raw Markdown text.
        """

        response = model.generate_content([prompt, document_part])

        in_tokens = response.usage_metadata.prompt_token_count
        out_tokens = response.usage_metadata.candidates_token_count

        return response.text, in_tokens, out_tokens

    except Exception as e:
        st.error(f"Error during LLM formatting: {e}")
        return None, 0, 0


@st.cache_data(show_spinner=False)
def validate_markdown(file_bytes, mime_type, draft_markdown, api_key, selected_model):
    """Secondary pass to check for hallucinations or transcription errors."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(selected_model)

        document_part = {
            "mime_type": mime_type,
            "data": file_bytes
        }

        prompt = f"""
        You are a strict QA Editor. I am providing you with the original PDF document and a drafted Markdown transcription of that document.

        Your job is to cross-check the Markdown draft against the original PDF and fix ANY errors.
        Pay special attention to:
        - Missing or altered minus signs in text and tables.
        - Misaligned or jumbled data in tables.
        - Abstracted variables where actual numbers should be.
        - Mangled mathematical formulas.

        Here is the draft Markdown to correct:
        ====================
        {draft_markdown}
        ====================

        Output ONLY the fully corrected Markdown text. Do not include any notes about what you changed.
        """

        response = model.generate_content([prompt, document_part])

        in_tokens = response.usage_metadata.prompt_token_count
        out_tokens = response.usage_metadata.candidates_token_count

        return response.text, in_tokens, out_tokens

    except Exception as e:
        st.error(f"Error during validation: {e}")
        return None, 0, 0


# --- MAIN UI ---
st.title("📄 PDF to E-Reader Markdown (Native Vision)")
st.write(
    "Upload a PDF. The AI will read the native file (preserving tables and math) and output a clean Markdown file.")

uploaded_file = st.file_uploader("Upload your PDF article", type=["pdf"])

if uploaded_file is not None:
    if not api_key:
        st.warning("👈 Please enter your API key in the sidebar to proceed.")
    else:
        if st.button("Convert to Markdown", type="primary"):

            with st.status("Processing Document...", expanded=True) as status:
                file_bytes = uploaded_file.getvalue()
                mime_type = "application/pdf"

                st.write(f"1️⃣ Sending native PDF to {model_name} for formatting...")
                markdown_result, in_tok, out_tok = format_pdf_to_markdown(file_bytes, mime_type, api_key, model_name)

                if markdown_result:
                    total_in = in_tok
                    total_out = out_tok

                    if run_validation:
                        st.write("2️⃣ Running strict QA Validation Pass...")
                        markdown_result, val_in, val_out = validate_markdown(file_bytes, mime_type, markdown_result,
                                                                             api_key, model_name)
                        total_in += val_in
                        total_out += val_out

                    status.update(label="Conversion Complete!", state="complete", expanded=False)

                    st.session_state['markdown'] = markdown_result
                    st.session_state['stats'] = (total_in, total_out)
                    st.session_state['filename'] = uploaded_file.name.replace('.pdf', '.md')
                else:
                    status.update(label="Failed at LLM stage.", state="error")

# --- RESULTS UI ---
if 'markdown' in st.session_state:
    # Safely unpack 3 values (assuming you saved total_cost as the 3rd item)
    in_tok, out_tok = st.session_state['stats']

    st.subheader("📊 Conversion Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Input Tokens", f"{in_tok:,}")
    col2.metric("Total Output Tokens", f"{out_tok:,}")
    #col3.metric("Estimated Cost", f"${cost:.5f}")

    st.divider()

    st.subheader("Preview")
    with st.expander("Click to view Markdown preview", expanded=True):
        st.markdown(st.session_state['markdown'])

    st.divider()

    filename = st.session_state.get('filename', 'converted_article.md')

    st.download_button(
        label="⬇️ Download Markdown File",
        data=st.session_state['markdown'],
        file_name=filename,
        mime="text/markdown",
        type="primary",
        use_container_width=True
    )