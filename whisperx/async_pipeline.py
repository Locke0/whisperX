import asyncio
import json
import torch
import numpy as np
from sortedcontainers import SortedDict
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from .vad import merge_chunks, load_vad_model
from .asr import load_model
from .alignment import load_align_model
from .utils import WriteJSON, get_writer
from .SubtitlesProcessor import SubtitlesProcessor

class WhisperXRealtimePipeline:
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16",
        vad_onset: float = 0.500,
        vad_offset: float = 0.363,
        batch_size: int = 1,
        chunk_size: int = 2,  # 2-second chunks for real-time
        no_align: bool = False,
        language: Optional[str] = None,
    ):
        # Initialize queues
        self.audio_chunks = asyncio.Queue()
        self.vad_results = asyncio.Queue()
        self.asr_results = asyncio.Queue()
        self.aligned_results = asyncio.Queue()

        # Processing pools
        self.asr_pool = ProcessPoolExecutor(max_workers=2)
        self.align_pool = ProcessPoolExecutor(max_workers=2)

        # Result management
        self.pending_results = SortedDict()  # Ordered by timestamp
        self.result_lock = asyncio.Lock()

        # Initialize models
        # TODO: double check with asr.py default options
        self.asr_options = {
            "without_timestamps": True,  # Faster processing
            "condition_on_previous_text": True,  # Better context
            "suppress_tokens": [-1],
        }
        
        self.model = load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            asr_options=self.asr_options,
            vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}
        )

        if not no_align:
            self.align_model, self.align_metadata = load_align_model(                
                language or "en",
                device
            )
        else:
            self.align_model = None



        self.chunk_size = chunk_size
        self.buffer = np.array([], dtype=np.float32)

        # TODO: check if this is needed
        self.previous_text = ""
        self.writer = WriteJSON(".")  # For consistent result formatting
        
    """
    1. audio_processor
    2. asr_processor
    3. align_processor
    4. output_processor

    process_stream()
    """


    # TODO: break this apart and modify this to process_stream()
    async def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process a single chunk of audio data"""
        self.buffer = np.append(self.buffer, audio_chunk)
        
        # Process if we have enough audio
        if len(self.buffer) >= self.chunk_size * 16000:  # 16kHz sample rate
            # Extract chunk for processing
            process_chunk = self.buffer[:self.chunk_size * 16000]
            self.buffer = self.buffer[self.chunk_size * 16000:]
            
            # Get VAD segments
            vad_segments = self.model.vad_model({
                "waveform": torch.from_numpy(process_chunk).unsqueeze(0),
                "sample_rate": 16000
            })
            
            # Merge VAD segments
            merged_segments = merge_chunks(
                vad_segments,
                self.chunk_size,
                onset=self.model._vad_params["vad_onset"],
                offset=self.model._vad_params["vad_offset"]
            )
            
            results = []
            if merged_segments:
                # Process segments
                result = self.model.transcribe(
                    process_chunk,
                    batch_size=1,  # Real-time processing
                    chunk_size=self.chunk_size,
                    initial_prompt=self.previous_text
                )
                
                if result["segments"]:
                    # Update context
                    self.previous_text = result["segments"][-1]["text"]
                    
                    # Process subtitles for better formatting
                    subtitle_processor = SubtitlesProcessor(
                        result["segments"],
                        result.get("language", "en"),
                        max_line_length=30  # Shorter for real-time display
                    )
                    formatted_segments = subtitle_processor.process_segments(
                        advanced_splitting=False  # Faster processing
                    )
                    
                    return {
                        "type": "transcription",
                        "segments": formatted_segments,
                        "text": self.previous_text
                    }
            
            return None


    async def process_audio_chunk(self, chunk: np.ndarray, timestamp: float):
        """Process incoming audio with overlapping windows"""
        # Reference: https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/audio/Basics.html
        window_size = int(self.processing_window * SAMPLE_RATE)
        overlap_size = int(self.overlap * SAMPLE_RATE)
        
        async with self.result_lock:
            sequence = self.next_sequence
            self.next_sequence += 1
            self.sequence_map[sequence] = {
                'timestamp': timestamp,
                'status': 'processing',
                'dependencies': set()
            }
            
            # If this chunk overlaps with previous, mark dependency
            if sequence > 0:
                self.sequence_map[sequence]['dependencies'].add(sequence - 1)

        return sequence

    async def vad_processor(self):
        """VAD processing with overlap handling"""
        while True:
            sequence, chunk = await self.audio_chunks.get()
            try:
                # Process in real-time thread as per guidelines
                # Reference: http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing
                vad_segments = self.vad_model({
                    "waveform": torch.from_numpy(chunk).unsqueeze(0),
                    "sample_rate": SAMPLE_RATE
                })
                await self.vad_results.put((sequence, (chunk, vad_segments)))
            finally:
                self.audio_chunks.task_done()

    async def asr_processor(self):
        """Parallel ASR processing"""
        while True:
            sequence, (chunk, vad_segments) = await self.vad_results.get()
            try:
                # Process in parallel pool
                result = await self.asr_pool.submit(
                    self.model.transcribe,
                    audio=chunk,
                    batch_size=1,
                    chunk_size=2
                )
                await self.asr_results.put((sequence, result))
            finally:
                self.vad_results.task_done()

    async def result_manager(self):
        """Ordered result assembly and delivery"""
        while True:
            try:
                async with self.result_lock:
                    # Check for completed sequences
                    ready_sequences = []
                    for seq, data in self.sequence_map.items():
                        if (data['status'] == 'complete' and 
                            not data['dependencies']):
                            ready_sequences.append(seq)
                    
                    # Process ready sequences in order
                    for seq in sorted(ready_sequences):
                        result = self.pending_results.pop(seq)
                        await self.send_result(result)
                        # Update dependencies
                        for next_seq in self.sequence_map:
                            if seq in self.sequence_map[next_seq]['dependencies']:
                                self.sequence_map[next_seq]['dependencies'].remove(seq)
                        
                await asyncio.sleep(0.01)  # Prevent busy-waiting
            except Exception as e:
                print(f"Error in result manager: {e}")


    async def handle_websocket_stream(self, websocket):
        """Handle WebSocket connection and audio streaming"""
        current_timestamp = 0.0
        buffer = np.array([], dtype=np.float32)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Convert audio bytes to numpy array
                    audio_chunk = np.frombuffer(message, dtype=np.float32)
                    
                    # Process chunk
                    result = await self.process_audio_chunk(audio_chunk)
                    if result:
                        await websocket.send(json.dumps(result))
                        
                    elif isinstance(message, str):
                        msg = json.loads(message)
                        if msg.get("type") == "end":
                            # Process any remaining audio
                            if len(self.buffer) > 0:
                                final_result = await self.process_audio_chunk(self.buffer)
                                if final_result:
                                    await websocket.send(json.dumps(final_result))
                            break
                            
        except Exception as e:
            print(f"Error in websocket handling: {e}")
            
        finally:
            # Cleanup
            self.buffer = np.array([], dtype=np.float32)
            self.previous_text = ""
