# utils.py

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import mediapipe as mp
import pickle
import faiss
from scipy.spatial.distance import cosine
import pyttsx3 # Added for Text-to-Speech
import threading # Added for TTS threading
import queue # Added for TTS message queue

# --- Configuration (Adjust as needed, these will be passed to RobotBrain) ---
EMBEDDING_DIM = 512
CANDIDATE_MATCH_THRESHOLD = 0.75 # Max distance to consider a match

# Set device globally for utility functions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load PyTorch Model (initialized once at application startup)
resnet_model = None
resnet_auto_transforms = None
face_detection_pipeline = None

def initialize_models(model_path):
    """Initializes the ResNet model and MediaPipe face detection pipeline."""
    global resnet_model, resnet_auto_transforms, face_detection_pipeline
    
    # ResNet Model
    from torchvision.models import resnet50, ResNet50_Weights
    resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
    resnet_auto_transforms = ResNet50_Weights.DEFAULT.transforms()
    resnet_model.fc = torch.nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=2048, out_features=EMBEDDING_DIM)
    ).to(DEVICE)

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ERROR: Model file not found at {model_path}.")
        resnet_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        resnet_model.eval()
        print(f"Custom embedding model weights loaded successfully from {model_path} to {DEVICE}.")
    except Exception as e:
        print(f"ERROR: Could not load custom embedding model weights from {model_path}: {e}")
        # In a FastAPI app, you might want to raise an exception here to prevent startup
        raise

    # MediaPipe Face Detection setup
    mp_face_detection = mp.solutions.face_detection
    face_detection_pipeline = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
    print("MediaPipe Face Detection pipeline initialized.")

def get_embedding_from_image_or_frame(img_input):
    """
    Gets the embedding from a PIL Image or OpenCV frame using the ResNet model.
    Assumes the input is a cropped face.
    Returns L2-normalized embedding (numpy array), or None if invalid.
    """
    global resnet_model, resnet_auto_transforms # Ensure global models are used
    if resnet_model is None or resnet_auto_transforms is None:
        print("ERROR: ResNet model or transforms not initialized.")
        return None

    try:
        if isinstance(img_input, np.ndarray): # If OpenCV frame (BGR)
            image = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        elif isinstance(img_input, Image.Image): # If PIL Image
            image = img_input.convert("RGB")
        else:
            return None

        face_tensor = resnet_auto_transforms(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding_tensor = resnet_model(face_tensor)
            normalized_embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
            
            if not torch.isfinite(normalized_embedding_tensor).all():
                return None

            embedding_np = normalized_embedding_tensor.cpu().numpy()[0]
            
            if embedding_np.shape[0] != EMBEDDING_DIM:
                return None

            return embedding_np
    except Exception as e:
        # print(f"  ERROR generating embedding: {e}") # Keep silent for cleaner output
        return None

def detect_face_in_frame(frame):
    """
    Detects the largest face in a given frame and returns its bounding box,
    the cropped face image for embedding, and the original frame.
    """
    global face_detection_pipeline # Ensure global pipeline is used
    if face_detection_pipeline is None:
        print("ERROR: MediaPipe face detection pipeline not initialized.")
        return None, None # Return original frame if pipeline not ready

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_detection_pipeline.process(rgb_frame)
    rgb_frame.flags.writeable = True

    largest_detection = None
    max_area = 0
    cropped_face_img_for_embedding = None
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                  int(bbox.width * w), int(bbox.height * h)
            
            x, y = max(0, x), max(0, y)
            width, height = min(w - x, width), min(h - y, height)

            current_area = width * height
            if current_area > max_area and width > 0 and height > 0:
                max_area = current_area
                largest_detection = (x, y, width, height)
                
                # Crop with margin for embedding
                margin_x = int(width * 0.1)
                margin_y = int(height * 0.1)
                crop_x1 = max(0, x - margin_x)
                crop_y1 = max(0, y - margin_y)
                crop_x2 = min(w, x + width + margin_x)
                crop_y2 = min(h, y + height + margin_y)
                cropped_face_img_for_embedding = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    return largest_detection, cropped_face_img_for_embedding

def load_known_faces(face_data_path, names_path):
    """
    Loads face embeddings and names, and initializes the FAISS index.
    Returns (all_face_embeddings_list, all_names_list, faiss_index, faiss_index_mapping).
    """
    loaded_embeddings_list = []
    loaded_names = []
    faiss_idx = None
    faiss_idx_map = []
    try:
        if not os.path.exists(face_data_path) or not os.path.exists(names_path):
            print(f"WARNING: Known faces pickle files not found at {face_data_path} or {names_path}. Starting with empty known faces data.")
            return [], [], None, []

        with open(face_data_path, 'rb') as f:
            loaded_embeddings_np = pickle.load(f)
        with open(names_path, 'rb') as f:
            loaded_names = pickle.load(f)

        if not isinstance(loaded_embeddings_np, np.ndarray) or loaded_embeddings_np.ndim != 2 or \
           loaded_embeddings_np.shape[1] != EMBEDDING_DIM:
            print(f"WARNING: Loaded embeddings from {face_data_path} have unexpected format/shape. Resetting.")
            return [], [], None, []

        if len(loaded_embeddings_np) != len(loaded_names):
            print(f"WARNING: Mismatch in counts between loaded embeddings ({len(loaded_embeddings_np)}) and names ({len(loaded_names)}). Data might be corrupted. Resetting known faces.")
            return [], [], None, []

        print(f"Known faces data loaded successfully from {face_data_path} and {names_path}. Loaded {len(loaded_embeddings_np)} entries.")
        loaded_embeddings_list = loaded_embeddings_np.tolist() # Keep as list for easy append later

        # Set up FAISS index after loading
        if loaded_embeddings_np.size > 0:
            embeddings_for_faiss = np.array(loaded_embeddings_np, dtype=np.float32)
            D = embeddings_for_faiss.shape[1] 
            faiss_idx = faiss.IndexFlatL2(D)
            faiss_idx.add(embeddings_for_faiss)
            faiss_idx_map = list(range(len(loaded_names)))
            print(f"FAISS index created and populated with {faiss_idx.ntotal} vectors.")
        else:
            print("No known faces to load, FAISS index not created.")

        return loaded_embeddings_list, loaded_names, faiss_idx, faiss_idx_map
    except pickle.UnpicklingError as e:
        print(f"ERROR: Could not unpickle data from files. Check if {face_data_path} and {names_path} are valid pickle files: {e}")
        return [], [], None, []
    except Exception as e:
        print(f"ERROR: Could not load known faces data or set up FAISS index: {e}")
        return [], [], None, []

def save_known_faces(embeddings, names, face_data_path, names_path):
    """
    Saves face embeddings and names. Does not return FAISS index,
    as it will be rebuilt on load or within RobotBrain on add.
    """
    try:
        if not embeddings:
            print("WARNING: No embeddings to save. Skipping saving known faces.")
            return False

        embeddings_to_save = np.array(embeddings, dtype=np.float32)
        with open(face_data_path, 'wb') as f:
            pickle.dump(embeddings_to_save, f)
        print(f"Known face embeddings saved to {face_data_path} (Shape: {embeddings_to_save.shape})")

        with open(names_path, 'wb') as f:
            pickle.dump(names, f)
        print(f"Known face names saved to {names_path} (Count: {len(names)})")
        
        return True
    except Exception as e:
        print(f"ERROR: Could not save known faces data: {e}")
        return False

def find_top_n_neighbors_faiss(new_embedding_np, faiss_idx, faiss_idx_map, known_names, max_distance_threshold, n=3):
    """
    Performs a FAISS nearest neighbor search and returns top N matches
    within a given distance threshold, sorted by distance.
    Returns a list of (distance, name) tuples.
    """
    if faiss_idx is None or faiss_idx.ntotal == 0:
        return []

    try:
        # Reshape new_embedding for FAISS search (1, D)
        query_embedding = np.array([new_embedding_np], dtype=np.float32)
        
        # Perform search: D = distances, I = indices
        distances, indices = faiss_idx.search(query_embedding, n)
        
        results = []
        # distances and indices are 2D arrays (batch_size, n)
        # We only have batch_size = 1
        for i in range(n):
            faiss_idx = indices[0][i]
            dist = distances[0][i]
            
            # Check if the index is valid (not -1, which indicates no match found for that slot)
            # and within the threshold
            if faiss_idx != -1 and dist <= max_distance_threshold:
                # Map FAISS internal index back to our original all_names index
                original_name_idx = faiss_idx_map[faiss_idx]
                name = known_names[original_name_idx]
                results.append((dist, name))
        
        # Sort by distance (ascending) and return top N unique names
        # FAISS already returns sorted by distance, but we re-sort to ensure uniqueness
        # if multiple embeddings map to the same person and we only want N unique people.
        unique_matches = []
        seen_names = set()
        for dist, name in results:
            if name not in seen_names:
                unique_matches.append((dist, name))
                seen_names.add(name)
                if len(unique_matches) >= n:
                    break
        return unique_matches
    except Exception as e:
        print(f"ERROR during FAISS search: {e}")
        return []

# --- Text-to-Speech Manager ---
class TTSManager:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            # You can set properties like voice, rate, volume here
            # self.engine.setProperty('rate', 150) # Speed of speech
            # self.engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)
            
            # Optional: Select a specific voice
            # voices = self.engine.getProperty('voices')
            # for voice in voices:
            #     print(f"Voice ID: {voice.id}, Name: {voice.name}, Languages: {voice.languages}")
            # self.engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0') # Example Windows voice

            self.message_queue = queue.Queue()
            self.speaker_thread = threading.Thread(target=self._speaker_loop, daemon=True)
            self.speaker_thread.start()
            print("TTSManager initialized and speaker thread started.")
        except Exception as e:
            self.engine = None
            print(f"WARNING: Could not initialize TTS engine: {e}. Speech will be disabled.")

    def _speaker_loop(self):
        """Dedicated thread for speaking messages."""
        while True:
            message = self.message_queue.get() # Blocks until a message is available
            if message is None: # Sentinel value to stop the thread
                break
            if self.engine:
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                    # If the queue is empty, stop the speaking immediately.
                    # This prevents the engine from waiting for more commands if there are none.
                    self.engine.stop() 
                except Exception as e:
                    print(f"ERROR: TTS engine failed to speak: {e}")
            self.message_queue.task_done()

    def say(self, message: str):
        """Adds a message to the queue to be spoken."""
        if self.engine:
            self.message_queue.put(message)
        else:
            print(f"TTS Disabled: {message}") # Fallback if engine failed to initialize

    def stop(self):
        """Stops the TTS speaker thread."""
        if self.engine:
            self.message_queue.put(None) # Send sentinel to stop thread
            self.speaker_thread.join(timeout=2)
            if self.speaker_thread.is_alive():
                print("WARNING: TTS speaker thread did not terminate gracefully.")

# Global TTSManager instance
tts_manager = TTSManager()

