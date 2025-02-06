import os
import argparse
import requests
import json
import yt_dlp

# For our custom diarization:
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# For splitting audio into chunks:
from pydub import AudioSegment

# ---------------------------
# Configuration / Tokens
# ---------------------------
# Set your OpenAI API key here or via the environment variable.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-your-key")

# Maximum allowed file size (in bytes) for the transcription API:
MAX_FILE_SIZE = 26214400  # 26 MB

# ---------------------------
# Function: Download Audio from YouTube
# ---------------------------
def download_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        # Use a local cache directory to avoid permission issues.
        'cachedir': os.path.join(os.getcwd(), 'yt_dlp_cache'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s',
        'paths': {'home': os.path.dirname(output_path)},
        'no_warnings': False,
        'quiet': False
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Change the extension from original to .mp3
            downloaded_file = os.path.join(
                os.path.dirname(output_path),
                ydl.prepare_filename(info).replace(f".{info['ext']}", ".mp3")
            )
            os.rename(downloaded_file, output_path)
            return True
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return False

# ---------------------------
# Function: Transcribe Audio via OpenAI's Whisper API (single file)
# ---------------------------
def transcribe_audio_via_http(audio_path, api_key):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "language": "en",
        "temperature": 0
    }
    with open(audio_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(url, headers=headers, data=data, files=files)
    
    print("Response status code:", response.status_code)
    raw_content = response.content.decode("utf-8", errors="replace")
    print("Response content:", raw_content)
    
    if response.status_code == 200:
        try:
            transcription_json = json.loads(raw_content)
            return transcription_json
        except Exception as e:
            print("Error parsing JSON:", e)
            return None
    else:
        return None

# ---------------------------
# Function: Transcribe Audio with Chunking
# ---------------------------
def transcribe_audio_with_chunking(audio_path, api_key, chunk_duration=60):
    """
    If the file size is larger than MAX_FILE_SIZE, split the audio into chunks
    (of chunk_duration seconds each) and transcribe them individually.
    The transcription segments from each chunk are then time-shifted and merged.
    """
    file_size = os.path.getsize(audio_path)
    if file_size <= MAX_FILE_SIZE:
        print("File size within allowed limit; transcribing whole file.")
        return transcribe_audio_via_http(audio_path, api_key)
    
    print("File size exceeds limit. Splitting audio into chunks...")
    audio = AudioSegment.from_file(audio_path)
    audio_duration_sec = len(audio) / 1000.0  # duration in seconds
    chunks = []
    start_times = []
    chunk_duration_ms = chunk_duration * 1000  # pydub works in milliseconds
    for start in range(0, len(audio), int(chunk_duration_ms)):
        chunk = audio[start:start+int(chunk_duration_ms)]
        chunks.append(chunk)
        start_times.append(start / 1000.0)  # record start time in seconds

    combined_segments = []
    combined_text = ""
    # Process each chunk separately.
    for i, chunk in enumerate(chunks):
        temp_chunk_file = f"temp_chunk_{i}.mp3"
        chunk.export(temp_chunk_file, format="mp3")
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        transcription = transcribe_audio_via_http(temp_chunk_file, api_key)
        if transcription is None:
            print(f"Transcription failed for chunk {i+1}.")
            os.remove(temp_chunk_file)
            continue

        # Adjust the segment times by adding the chunk's start time.
        for seg in transcription.get("segments", []):
            seg["start"] += start_times[i]
            seg["end"] += start_times[i]
            combined_segments.append(seg)
        combined_text += transcription.get("text", "") + " "
        os.remove(temp_chunk_file)
    
    # Build a combined transcription JSON structure.
    combined_transcription = {
        "task": "transcribe",
        "language": transcription.get("language", "en") if transcription else "en",
        "duration": audio_duration_sec,
        "text": combined_text.strip(),
        "segments": sorted(combined_segments, key=lambda x: x["start"])
    }
    return combined_transcription

# ---------------------------
# Function: Perform Diarization Using Librosa & KMeans
# ---------------------------
def get_diarization_librosa(audio_path, n_clusters=2, hop_length=512):
    """
    Load the audio, compute MFCC features, scale them, cluster the frames
    using KMeans, and group consecutive frames with the same cluster label into segments.
    
    Returns:
        A list of dictionaries, each with keys: 'start', 'end', 'speaker'
    """
    # Load audio using librosa
    audio, sr = librosa.load(audio_path, sr=None)
    # Compute MFCC features; shape is (n_mfcc, n_frames)
    mfccs = librosa.feature.mfcc(audio, sr=sr, hop_length=hop_length)
    
    # Scale (transpose so that each row is a frame)
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs.T)
    
    # Perform KMeans clustering on the frames
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    speaker_labels = kmeans.fit_predict(mfccs_scaled)
    
    # Get the time for each frame
    frame_times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr, hop_length=hop_length)
    
    # Group consecutive frames with the same label into segments.
    segments = []
    current_label = speaker_labels[0]
    seg_start = frame_times[0]
    
    for i in range(1, len(speaker_labels)):
        if speaker_labels[i] != current_label:
            seg_end = frame_times[i]
            segments.append({'start': seg_start, 'end': seg_end, 'speaker': f"Speaker {current_label}"})
            current_label = speaker_labels[i]
            seg_start = frame_times[i]
    # Append the final segment
    segments.append({'start': seg_start, 'end': frame_times[-1], 'speaker': f"Speaker {current_label}"})
    
    # Debug print of each diarization segment
    for seg in segments:
        print(f"Diarization segment: {seg['start']:.2f} to {seg['end']:.2f} - {seg['speaker']}")
    
    return segments

# ---------------------------
# Function: Build Structured Report for Speaker Assignment
# ---------------------------
def build_structured_report(transcription_json, diarization_segments):
    """
    For each transcription segment, compute the overlap with diarization segments,
    assign the speaker based on maximum overlap, and build a structured report.
    
    Returns:
        A list of dictionaries with keys: 'start', 'end', 'speaker', 'text'
    """
    segments = transcription_json.get("segments", [])
    report = []
    
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_text = seg.get("text", "").strip()
        
        speaker_overlap = {}
        for d in diarization_segments:
            d_start = d['start']
            d_end = d['end']
            label = d['speaker']
            # Calculate overlap duration
            overlap = max(0, min(seg_end, d_end) - max(seg_start, d_start))
            if overlap > 0:
                speaker_overlap[label] = speaker_overlap.get(label, 0) + overlap
        
        assigned_speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else "Unknown"
        report.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": assigned_speaker,
            "text": seg_text
        })
    
    return report

# ---------------------------
# Function: Convert Structured Report to Plain Text Transcript
# ---------------------------
def report_to_plain_text(report):
    """
    Convert the structured report (list of dicts) into a plain text transcript.
    """
    transcript_lines = []
    for seg in report:
        transcript_lines.append(f"{seg['speaker']}: {seg['text']}")
    return "\n".join(transcript_lines)

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate transcript with speaker diarization from a YouTube video")
    parser.add_argument("url", help="YouTube video URL")
    args = parser.parse_args()

    # Create a temporary directory for audio files
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_path = os.path.join(temp_dir, "audio.mp3")
    
    try:
        print("Downloading audio...")
        if download_audio(args.url, audio_path):
            print(f"Audio downloaded successfully to: {audio_path}")
            
            if os.path.exists(audio_path):
                print("Transcribing audio using HTTP POST to OpenAI API (with chunking if needed)...")
                # Use the chunking version so that large files are split
                transcription_json = transcribe_audio_with_chunking(audio_path, OPENAI_API_KEY, chunk_duration=60)
                
                if transcription_json:
                    try:
                        print("Running speaker diarization with Librosa + KMeans...")
                        diarization_segments = get_diarization_librosa(audio_path, n_clusters=2, hop_length=512)
                    except Exception as e:
                        print("Speaker diarization failed:", e)
                        diarization_segments = None
                        
                    if diarization_segments:
                        structured_report = build_structured_report(transcription_json, diarization_segments)
                        final_transcript = report_to_plain_text(structured_report)
                    else:
                        final_transcript = (
                            "Speaker diarization was not performed. Transcription:\n" +
                            " ".join(seg.get("text", "").strip() for seg in transcription_json.get("segments", []))
                        )
                        structured_report = None
                    
                    print("\nFinal Transcript:")
                    print(final_transcript)
                    
                    # Save plain transcript to file
                    transcript_file = os.path.join(os.getcwd(), "transcript.txt")
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(final_transcript)
                    print(f"\nTranscript saved to: {transcript_file}")
                    
                    # Save structured report to JSON file if available
                    if structured_report is not None:
                        report_file = os.path.join(os.getcwd(), "report.json")
                        with open(report_file, "w", encoding="utf-8") as f:
                            json.dump(structured_report, f, indent=4)
                        print(f"Structured report saved to: {report_file}")
                else:
                    print("Transcription failed.")
            else:
                print(f"Error: Audio file not found at {audio_path}")
        else:
            print("Failed to download audio")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Cleanup: remove the temporary audio file and directory
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as ex:
                print(f"Cleanup error: {ex}")
        try:
            os.rmdir(temp_dir)
        except Exception as ex:
            print(f"Cleanup error: {ex}")

if __name__ == "__main__":
    main()
