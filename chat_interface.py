import requests
import streamlit as st
from typing import Dict, List
import json
import iso639
import langdetect

class TranscriptChat:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.context_window = 5  # Number of previous exchanges to maintain context
        self.supported_languages = self._get_supported_languages()

    def _get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages"""
        languages = {}
        for lang in iso639.languages:
            try:
                languages[lang.name.lower()] = lang.part1
            except AttributeError:
                continue
        return languages

    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            detected = langdetect.detect(text)
            return detected
        except:
            return 'en'  # Default to English if detection fails

    def _prepare_context(self, transcript: str) -> Dict[str, List[str]]:
        """Prepare searchable context from transcript"""
        # Split transcript into meaningful chunks
        chunks = []
        for line in transcript.split('\n'):
            if ':' in line:
                speaker, text = line.split(':', 1)
                chunks.append({
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'combined': line.strip()
                })
        return {'chunks': chunks}

    def _get_relevant_context(self, prompt: str, transcript_context: Dict, n_chunks: int = 3) -> str:
        """Get most relevant transcript chunks for the question"""
        # Simple keyword matching (can be improved with embeddings)
        keywords = prompt.lower().split()
        chunk_scores = []
        
        for chunk in transcript_context['chunks']:
            score = sum(1 for keyword in keywords if keyword in chunk['combined'].lower())
            chunk_scores.append((score, chunk['combined']))
        
        # Get top N most relevant chunks
        relevant_chunks = sorted(chunk_scores, reverse=True)[:n_chunks]
        return "\n".join(chunk[1] for chunk in relevant_chunks if chunk[0] > 0)

    def _enhance_prompt(self, prompt: str, conversation_history: List[Dict]) -> str:
        """Create an enhanced prompt with conversation context"""
        # Get recent conversation context
        recent_context = conversation_history[-self.context_window:] if conversation_history else []
        
        enhanced_prompt = f"""Based on the transcript excerpts provided, please answer the following question.
        Remember to:
        1. Only use information explicitly stated in the transcript
        2. If there are time stamps or speaker identifications, include them
        3. If the answer requires synthesizing information from multiple parts, explain the connection
        4. If the information is not in the transcript, say so clearly
        
        Recent conversation context:
        {' | '.join([f"{m['role']}: {m['content']}" for m in recent_context])}
        
        Current question: {prompt}
        """
        return enhanced_prompt

    def query(self, prompt: str, transcript: str, conversation_history: List[Dict] = None) -> str:
        """Enhanced multilingual query processing"""
        try:
            # Detect prompt language
            prompt_lang = self._detect_language(prompt)
            
            # Prepare context and prompts
            transcript_context = self._prepare_context(transcript)
            relevant_context = self._get_relevant_context(prompt, transcript_context)
            enhanced_prompt = self._enhance_prompt(prompt, conversation_history or [])
            
            # Create language-specific system prompt
            system_prompt = self._get_system_prompt(prompt_lang)

            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nTranscript context:\n{relevant_context}\n\nQuestion ({prompt_lang}): {prompt}\n\nPlease respond in the same language as the question.",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return self._format_response(response.json().get('response', ''), prompt_lang)
            return f"Error: Server returned status code {response.status_code}"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _get_system_prompt(self, lang: str) -> str:
        """Get language-specific system prompt"""
        prompts = {
            'en': """You are a professional transcript analyst. Respond in English.
                    Focus on providing specific, evidence-based answers from the transcript.""",
            'fr': """Vous êtes un analyste professionnel de transcription. Répondez en français.
                    Concentrez-vous sur des réponses précises basées sur la transcription.""",
            'de': """Sie sind ein professioneller Transkriptanalyst. Antworten Sie auf Deutsch.
                    Konzentrieren Sie sich auf spezifische, evidenzbasierte Antworten aus dem Transkript.""",
            'es': """Eres un analista profesional de transcripciones. Responde en español.
                    Céntrate en proporcionar respuestas específicas basadas en la transcripción."""
            # Add more languages as needed
        }
        return prompts.get(lang, prompts['en'])

    def _format_response(self, response: str, lang: str) -> str:
        """Format response with language-specific formatting"""
        import re
        
        # Clean up the response
        response = self._clean_response(response)
        
        # Add markdown formatting for quotes
        response = re.sub(r'(?m)^>?\s*"([^"]+)"', r'> \1', response)
        
        # Format lists consistently
        response = re.sub(r'(?m)^[-*]\s', '• ', response)
        
        # Add bold to speaker names
        response = re.sub(r'(Speaker \d+|Unknown):', r'**\1**:', response)
        
        # Add language-specific formatting if needed
        if lang != 'en':
            response = f"[{lang.upper()}]\n{response}"
        
        return response

    def _clean_response(self, response):
        """Clean up the response to remove thinking process and format properly"""
        # Remove any text between <think> tags
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Clean up multiple newlines
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        return response.strip()

def render_chat_interface(transcript):
    """Enhanced multilingual chat interface"""
    st.subheader("Chat with your Transcript")
    
    # Initialize chat state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_bot' not in st.session_state:
        st.session_state.chat_bot = TranscriptChat()
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'en'
    
    # Language selector
    with st.sidebar:
        st.markdown("### Language Settings")
        available_languages = {
            'English': 'en',
            'Français': 'fr',
            'Deutsch': 'de',
            'Español': 'es',
            # Add more languages as needed
        }
        selected_lang_name = st.selectbox(
            "Select Interface Language",
            options=list(available_languages.keys()),
            index=0
        )
        st.session_state.selected_language = available_languages[selected_lang_name]
    
    # Example questions in selected language
    with st.expander("Example Questions"):
        examples = {
            'en': [
                "Who are the speakers in this conversation?",
                "What are the main topics discussed?",
                "Can you summarize what was said about [topic]?"
            ],
            'fr': [
                "Qui sont les intervenants dans cette conversation ?",
                "Quels sont les principaux sujets discutés ?",
                "Pouvez-vous résumer ce qui a été dit sur [sujet] ?"
            ],
            'de': [
                "Wer sind die Sprecher in diesem Gespräch?",
                "Was sind die Hauptthemen der Diskussion?",
                "Können Sie zusammenfassen, was über [Thema] gesagt wurde?"
            ],
            'es': [
                "¿Quiénes son los hablantes en esta conversación?",
                "¿Cuáles son los principales temas discutidos?",
                "¿Puede resumir lo que se dijo sobre [tema]?"
            ]
        }
        for example in examples.get(st.session_state.selected_language, examples['en']):
            st.markdown(f"- {example}")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Input placeholders in different languages
    input_placeholders = {
        'en': "Ask a question about the transcript",
        'fr': "Posez une question sur la transcription",
        'de': "Stellen Sie eine Frage zum Transkript",
        'es': "Haga una pregunta sobre la transcripción"
    }
    
    if prompt := st.chat_input(
        input_placeholders.get(st.session_state.selected_language, input_placeholders['en'])
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.chat_bot.query(
                    prompt, 
                    transcript,
                    st.session_state.messages
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
