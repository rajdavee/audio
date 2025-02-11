import os
import streamlit as st
import json
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import whisper
import torch
import os.path
from pathlib import Path
import base64
from sklearn.metrics import silhouette_score
import uuid
import shutil

# Set page config
st.set_page_config(page_title="Audio Transcription App", layout="wide")

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav', 'm4a', 'ogg', 'wma', 'aac', 'flac', 'aiff', 'ape', 'wv']

# Load Whisper model globally
@st.cache_resource
def load_whisper_model(model_size='base'):
    """Load and cache the Whisper model"""
    return whisper.load_model(model_size)

def transcribe_audio_local(audio_path, model):
    """Transcribe audio using local Whisper model"""
    try:
        result = model.transcribe(audio_path)
        return {
            "task": "transcribe",
            "language": result.get("language", "en"),
            "duration": result.get("duration", 0),
            "text": result.get("text", ""),
            "segments": result.get("segments", [])
        }
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None

def transcribe_audio_with_chunking(audio_path, model, chunk_duration=60):
    """Transcribe audio in chunks using local Whisper model"""
    audio = AudioSegment.from_file(audio_path)
    audio_duration_sec = len(audio) / 1000.0
    chunks = []
    start_times = []
    chunk_duration_ms = chunk_duration * 1000
    
    for start in range(0, len(audio), int(chunk_duration_ms)):
        chunk = audio[start:start+int(chunk_duration_ms)]
        chunks.append(chunk)
        start_times.append(start / 1000.0)

    combined_segments = []
    combined_text = ""
    
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        temp_chunk_file = f"temp_chunk_{i}.mp3"
        chunk.export(temp_chunk_file, format="mp3")
        st.info(f"Transcribing chunk {i+1}/{len(chunks)}...")
        
        transcription = transcribe_audio_local(temp_chunk_file, model)
        if transcription is None:
            st.warning(f"Transcription failed for chunk {i+1}.")
            os.remove(temp_chunk_file)
            continue

        for seg in transcription.get("segments", []):
            seg["start"] += start_times[i]
            seg["end"] += start_times[i]
            combined_segments.append(seg)
        combined_text += transcription.get("text", "") + " "
        os.remove(temp_chunk_file)
        progress_bar.progress((i + 1) / len(chunks))

    return {
        "task": "transcribe",
        "language": "en",
        "duration": audio_duration_sec,
        "text": combined_text.strip(),
        "segments": sorted(combined_segments, key=lambda x: x["start"])
    }

def convert_to_mp3(input_path, output_path):
    """Convert any audio format to MP3"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="mp3")
        return True
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return False

def get_diarization_librosa(audio_path, n_clusters=2, hop_length=512):
    """Improved speaker diarization using librosa and KMeans"""
    try:
        # Load audio with appropriate sample rate
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract more robust features
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr,
            n_mfcc=20,  # Increase number of MFCCs
            hop_length=hop_length,
            n_fft=2048,
            win_length=2048
        )
        
        # Add delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        combined_features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
        
        # Transpose and scale features
        features_scaled = StandardScaler().fit_transform(combined_features.T)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        speaker_labels = kmeans.fit_predict(features_scaled)
        
        # Get frame times
        frame_times = librosa.frames_to_time(
            np.arange(len(speaker_labels)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Group consecutive frames with the same label
        segments = []
        current_label = speaker_labels[0]
        seg_start = frame_times[0]
        min_segment_length = 1.0  # Minimum segment length in seconds
        
        for i in range(1, len(speaker_labels)):
            if speaker_labels[i] != current_label:
                seg_end = frame_times[i]
                # Only add segment if it's longer than minimum length
                if (seg_end - seg_start) >= min_segment_length:
                    segments.append({
                        'start': float(seg_start),
                        'end': float(seg_end),
                        'speaker': f"Speaker {current_label}"
                    })
                current_label = speaker_labels[i]
                seg_start = frame_times[i]
        
        # Add final segment
        if (frame_times[-1] - seg_start) >= min_segment_length:
            segments.append({
                'start': float(seg_start),
                'end': float(frame_times[-1]),
                'speaker': f"Speaker {current_label}"
            })
        
        # Merge very short segments
        merged_segments = []
        merge_threshold = 0.5  # seconds
        
        for i, seg in enumerate(segments):
            if i > 0 and seg['speaker'] == merged_segments[-1]['speaker'] and \
               (seg['start'] - merged_segments[-1]['end']) < merge_threshold:
                merged_segments[-1]['end'] = seg['end']
            else:
                merged_segments.append(seg.copy())
        
        return merged_segments
    
    except Exception as e:
        st.error(f"Diarization error: {str(e)}")
        return None

def build_structured_report(transcription_json, diarization_segments):
    """Improved structured report building with better speaker assignment"""
    if not diarization_segments:
        return None
        
    segments = transcription_json.get("segments", [])
    report = []
    
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_text = seg.get("text", "").strip()
        
        # Find the most overlapping speaker
        speaker_overlap = {}
        for d in diarization_segments:
            overlap = max(0, min(seg_end, d['end']) - max(seg_start, d['start']))
            if overlap > 0:
                speaker_overlap[d['speaker']] = speaker_overlap.get(d['speaker'], 0) + overlap
        
        # Assign speaker based on maximum overlap
        assigned_speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0] if speaker_overlap else "Unknown"
        
        report.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": assigned_speaker,
            "text": seg_text
        })
    
    return report

def report_to_plain_text(report):
    """Convert structured report to plain text format"""
    if not report:
        return ""
    transcript_lines = []
    for seg in report:
        transcript_lines.append(f"{seg['speaker']}: {seg['text']}")
    return "\n".join(transcript_lines)

def set_custom_style():
    """Set custom CSS styles"""
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        .stButton>button {
            background-color: #009688;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #00796b;
        }
        .output-box {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .app-header {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            background-color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .app-title {
            color: #1e88e5;
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Add logo path
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo-rag-lu-870x435.png")

def estimate_number_of_speakers(audio_path):
    """Automatically estimate the optimal number of speakers"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Calculate silhouette scores for different numbers of speakers
        best_score = -1
        optimal_speakers = 2  # default fallback
        
        # Try different numbers of speakers (2 to 5 is a reasonable range)
        for n in range(2, 6):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(mfccs.T)
            score = silhouette_score(mfccs.T, labels)
            
            if score > best_score:
                best_score = score
                optimal_speakers = n
        
        return optimal_speakers
    except Exception as e:
        st.warning(f"Could not estimate speakers automatically: {str(e)}")
        return 2  # fallback to default

def app_header():
    """Display app header with logo"""
    try:
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as f:
                logo_content = f.read()
                logo_base64 = base64.b64encode(logo_content).decode()
            st.markdown(f"""
                <div class="app-header">
                    <img src="data:image/png;base64,{logo_base64}" 
                         style="height: 60px; margin-right: 1rem;">
                    <h1 class="app-title">RAG.lu</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Logo not found at: {LOGO_PATH}")
            st.markdown("""
                <div class="app-header">
                    <h1 class="app-title">RAG.lu</h1>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading logo: {str(e)}")

from chat_interface import render_chat_interface

def main():
    set_custom_style()
    app_header()
    
    # Initialize session state for transcript and structured report
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'structured_report' not in st.session_state:
        st.session_state.structured_report = None
    if 'transcription_complete' not in st.session_state:
        st.session_state.transcription_complete = False

    with st.sidebar:
        st.markdown("### Model Settings")
        model_size = st.selectbox(
            "Select Model Size",
            ['tiny', 'base', 'small', 'medium', 'large'],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        st.markdown("### Advanced Settings")
        auto_detect = st.checkbox("Auto-detect number of speakers", value=True)
        if not auto_detect:
            n_speakers = st.number_input(
                "Number of Speakers",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                help="Manual override for number of speakers"
            )
    
    # Load the selected model
    with st.spinner(f"Loading Whisper {model_size} model..."):
        model = load_whisper_model(model_size)
    
    # Only show file uploader if transcription is not complete
    if not st.session_state.transcription_complete:
        uploaded_file = st.file_uploader("Upload an audio file", type=SUPPORTED_AUDIO_FORMATS)
        
        if uploaded_file is not None:
            # Create unique session ID for this upload
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            
            # Create temp directory with session ID
            temp_dir = os.path.join(os.getcwd(), "temp_audio", st.session_state.session_id)
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Clean up any existing files
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                
                # Save and convert file
                original_path = os.path.join(temp_dir, f"original.{uploaded_file.name.split('.')[-1]}")
                mp3_path = os.path.join(temp_dir, "audio.mp3")
                
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Convert to MP3 if needed
                if original_path.lower().endswith('.mp3'):
                    shutil.copy2(original_path, mp3_path)  # Use copy instead of rename
                    conversion_success = True
                else:
                    conversion_success = convert_to_mp3(original_path, mp3_path)
                
                if os.path.exists(original_path) and original_path != mp3_path:
                    os.remove(original_path)
                
                if conversion_success:
                    st.success("File uploaded and converted successfully!")
                    
                    if st.button("Start Transcription"):
                        try:
                            with st.spinner("Transcribing audio..."):
                                transcription_json = transcribe_audio_with_chunking(mp3_path, model, chunk_duration=60)
                            
                            if transcription_json:
                                with st.spinner("Detecting number of speakers..."):
                                    if auto_detect:
                                        n_speakers = estimate_number_of_speakers(mp3_path)
                                        st.info(f"Detected {n_speakers} speakers in the audio")
                                    
                                with st.spinner("Running speaker diarization..."):
                                    try:
                                        diarization_segments = get_diarization_librosa(mp3_path, n_clusters=n_speakers, hop_length=512)
                                    except Exception as e:
                                        st.error(f"Speaker diarization failed: {e}")
                                        diarization_segments = None
                                
                                if diarization_segments:
                                    structured_report = build_structured_report(transcription_json, diarization_segments)
                                    final_transcript = report_to_plain_text(structured_report)
                                    
                                    # Store in session state
                                    st.session_state.transcript = final_transcript
                                    st.session_state.structured_report = structured_report
                                    st.session_state.transcription_complete = True
                                    
                                    # Rerun to refresh the page with new state
                                    st.experimental_rerun()
                                    
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Cleanup on error
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return
            
            finally:
                # Cleanup temp files after processing
                if 'transcription_complete' in st.session_state and st.session_state.transcription_complete:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    if 'session_id' in st.session_state:
                        del st.session_state.session_id
    
    # Show results and chat interface if transcription is complete
    if st.session_state.transcription_complete:
        # Add a button to start over
        if st.sidebar.button("Start New Transcription"):
            st.session_state.transcript = None
            st.session_state.structured_report = None
            st.session_state.transcription_complete = False
            # Clear chat history
            if 'messages' in st.session_state:
                del st.session_state.messages
            st.experimental_rerun()
            
        st.subheader("Transcript:")
        st.text_area("Full Transcript", st.session_state.transcript, height=300)
        
        # Download buttons
        transcript_bytes = st.session_state.transcript.encode()
        st.download_button(
            label="Download Transcript",
            data=transcript_bytes,
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        if st.session_state.structured_report:
            json_str = json.dumps(st.session_state.structured_report, indent=4)
            st.download_button(
                label="Download Structured Report",
                data=json_str,
                file_name="report.json",
                mime="application/json"
            )
        
        # Add chat interface
        st.markdown("---")
        render_chat_interface(st.session_state.transcript)

def cleanup_temp_files():
    """Clean up all temporary files when the app exits"""
    temp_base_dir = os.path.join(os.getcwd(), "temp_audio")
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir)

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_temp_files()
