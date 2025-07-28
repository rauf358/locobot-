# robot_brain.py

import time
import os
import cv2
import numpy as np
import base64 # For handling image data
from PIL import Image # For image manipulation
import io # For handling image bytes
import json # For structured responses
import faiss
# --- NEW IMPORTS FOR EMAIL SENDING ---
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
# --- END NEW IMPORTS ---

from dotenv import load_dotenv # For loading environment variables from .env file
load_dotenv()

# Import utility functions for face detection, embedding, and FAISS, including TTSManager
from utils2 import (
    initialize_models,
    get_embedding_from_image_or_frame,
    detect_face_in_frame,
    load_known_faces,
    save_known_faces,
    find_top_n_neighbors_faiss,
    EMBEDDING_DIM, # Import EMBEDDING_DIM from utils for consistency
    tts_manager # Import the global TTSManager instance
)

# --- Robot State Definitions (Matching Workflow Diagram Logic) ---
ROBOT_STATE_IDLE = "IDLE" # Start state, waiting for visitor
ROBOT_STATE_DETECTING_FACE = "DETECTING_FACE" # Visitor detected, timing for 3s
ROBOT_STATE_PROCESSING_RECOGNITION = "PROCESSING_RECOGNITION" # Processing face for known/unknown
ROBOT_STATE_CONFIRMING_IDENTITY = "CONFIRMING_IDENTITY" # Asking "Are you Mr./Ms. ABC?"
ROBOT_STATE_GREET_UNKNOWN = "GREET_UNKNOWN" # Initial greeting for unknown visitor
ROBOT_STATE_ENROLL_NAME = "ENROLL_NAME" # Prompting unknown visitor for name
ROBOT_STATE_ENROLL_PHOTO_CONFIRM = "ENROLL_PHOTO_CONFIRM" # Confirming captured photo for enrollment
ROBOT_STATE_ASK_WHOM_TO_MEET = "ASK_WHOM_TO_MEET" # Asking "Whom would you like to meet?"
ROBOT_STATE_CONFIRM_WHO_TO_MEET = "CONFIRM_WHO_TO_MEET" # Confirming the name of the person to meet
ROBOT_STATE_CONTACTING_OFFICIAL = "CONTACTING_OFFICIAL" # Sending notification to official
ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE = "WAITING_FOR_OFFICIAL_RESPONSE" # Waiting for official's response
ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS = "OFFICIAL_UNREACHABLE_OPTIONS" # Official unreachable, offering wait/message options
ROBOT_STATE_LEAVE_MESSAGE = "LEAVE_MESSAGE" # Visitor opting to leave a message
ROBOT_STATE_WELCOMED = "WELCOMED" # Visitor successfully processed and welcomed (awaiting departure)
ROBOT_STATE_GOODBYE = "GOODBYE" # Visitor has left or interaction ended

# --- Constants ---
PRESENCE_DURATION_THRESHOLD_SEC = 3 # Time a face must be stable for initial trigger
LOSS_OF_RECOGNITION_THRESHOLD_FRAMES = 30 # Number of frames to tolerate loss of tracking before resetting
CANDIDATE_MATCH_THRESHOLD = 0.75 # Max distance for FAISS to consider a match
MAX_CANDIDATES_TO_PRESENT = 3 # Number of guesses for confirmation (for both visitor and official)
OFFICIAL_RESPONSE_TIMEOUT_SEC = 3 * 60 # 3 minutes for official response
MAX_OFFICIAL_CONTACT_ITERATIONS = 3 # Max retries for contacting official

# --- EMAIL CONFIGURATION (Read from Environment Variables for Security) ---
# For Gmail, you might need to enable "App Passwords" if 2FA is on.
# For other providers, check their SMTP settings.
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com") # e.g., "smtp.gmail.com" for Gmail
SMTP_PORT = int(os.getenv("SMTP_PORT", 587)) # 587 for TLS, 465 for SSL
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") # Or os.getenv("SENDER_PASSWORD") if you standardize
# Your email password/app password
# --- END EMAIL CONFIGURATION ---


class RobotBrain:
    def __init__(self, base_project_folder: str, model_path: str, face_data_path: str, names_path: str):
        self.base_project_folder = base_project_folder
        self.model_path = model_path
        self.face_data_path = face_data_path
        self.names_path = names_path

        # Initialize ML models (ResNet, MediaPipe) once at startup
        initialize_models(self.model_path)

        # Load known faces and FAISS index
        self.all_face_embeddings, self.all_names, self.faiss_index, self.faiss_index_mapping = \
            load_known_faces(self.face_data_path, self.names_path)
        
        # Placeholder for company officials' data (name: contact_info)
        # In a real system, this would be loaded from a database
        self.company_officials = {
            "misbah": {"email": "syedmisbah588@gmail.com", "phone": "555-123-4567"},
            "detective":{"email": "detective2k18@gmail.com", "phone": "033133349338"}
        }

        # --- Robot State Variables (persisted across API calls for a session) ---
        self.robot_state = ROBOT_STATE_IDLE
        self.face_present_start_time = None
        self.last_face_embedding = None # Embedding of the last detected face
        self.last_face_image_cv2 = None # OpenCV image of the last detected face (for sending to official)
        self.active_recognized_person = None # Name of the visitor currently interacting
        self.loss_of_recognition_counter = 0 # For tracking if a recognized person leaves
        
        # For visitor identity confirmation flow
        self.current_identity_candidates = [] # List of (distance, name) tuples for "Is this you?"
        self.current_identity_candidate_index = 0

        # For "whom to meet" flow
        self.visitor_spoken_name_raw = None # Raw input from STT for visitor's name
        self.visitor_confirmed_name = None # Confirmed name of the visitor
        self.whom_to_meet_spoken_name_raw = None # Raw input from STT for whom to meet
        self.whom_to_meet_confirmed_name = None # Confirmed name of the person to meet
        self.whom_to_meet_candidates = [] # List of (distance, name) tuples for "Whom to meet?"
        self.whom_to_meet_candidate_index = 0

        # For official contact flow
        self.official_contact_start_time = None
        self.official_contact_iteration = 0 # How many times we've tried to contact the official

        print("RobotBrain initialized.")

    def _reset_state(self):
        """Resets all relevant state variables to initial IDLE state."""
        self.robot_state = ROBOT_STATE_IDLE
        self.face_present_start_time = None
        self.last_face_embedding = None
        self.last_face_image_cv2 = None
        self.active_recognized_person = None
        self.loss_of_recognition_counter = 0
        self.current_identity_candidates = []
        self.current_identity_candidate_index = 0
        self.visitor_spoken_name_raw = None
        self.visitor_confirmed_name = None
        self.whom_to_meet_spoken_name_raw = None
        self.whom_to_meet_confirmed_name = None
        self.whom_to_meet_candidates = []
        self.whom_to_meet_candidate_index = 0
        self.official_contact_start_time = None
        self.official_contact_iteration = 0
        print("[SYSTEM] Robot state reset to IDLE.")
        return self._create_response("Robot is IDLE, waiting for a visitor.", "display_message", ROBOT_STATE_IDLE)

    def _create_response(self, message: str, action: str, state: str, options: list = None, prompt: str = None, image_base64: str = None):
        """
        Helper to create a consistent API response structure and trigger TTS.
        """
        # Speak the message whenever a response is created
        tts_manager.say(message)
        print(f"\nROBOT: {message}") # Keep console print for logging/debugging

        return {
            "message": message,
            "action": action,
            "state": state,
            "options": options,
            "prompt": prompt,
            "image_base64": image_base64
        }

    def _send_notification_to_official(self, official_name: str, visitor_name: str, photo_cv2_frame: np.ndarray = None, message_type: str = "visitor_arrival"):
        """
        Sends an actual email notification to a company official.
        """
        official_info = self.company_officials.get(official_name.lower())
        if not official_info or not official_info.get("email"):
            print(f"WARNING: Official '{official_name}' not found or no email address. Cannot send email.")
            return False

        recipient_email = official_info["email"]
        subject = ""
        body = ""
        
        if message_type == "visitor_arrival":
            subject = f"Visitor Alert: {visitor_name} is here to see {official_name}"
            body = f"Hello {official_name},\n\nA visitor named {visitor_name} is at the reception to meet you. Please respond via the app (approve/deny) or directly attend to them."
        elif message_type == "visitor_message":
            subject = f"Message from Visitor {visitor_name} for {official_name}"
            body = f"Hello {official_name},\n\nVisitor {self.active_recognized_person} has left a message for you: \"{visitor_name}\"." # visitor_name here is the message content
        elif message_type == "unreachable_info":
             subject = f"Visitor Inquiry: {visitor_name} is waiting for {official_name}"
             body = f"Hello {official_name},\n\nVisitor {visitor_name} is still waiting for you at reception. They were informed you are unreachable after multiple attempts. Please respond via the app or attend to them."

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if photo_cv2_frame is not None:
            try:
                # Convert OpenCV frame to JPEG bytes
                _, img_encoded = cv2.imencode('.jpg', photo_cv2_frame)
                img_bytes = img_encoded.tobytes()

                image = MIMEImage(img_bytes, _subtype="jpeg")
                image.add_header('Content-Disposition', 'attachment', filename=f'{visitor_name}_photo.jpg')
                msg.attach(image)
            except Exception as e:
                print(f"ERROR: Could not attach photo to email: {e}")

        try:
            print(f"\n--- ATTEMPTING TO SEND EMAIL ---")
            print(f"From: {SENDER_EMAIL}")
            print(f"To: {recipient_email}")
            print(f"Subject: {subject}")
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls() # Enable TLS encryption
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
            
            print(f"--- EMAIL SENT SUCCESSFULLY to {recipient_email} ---")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"ERROR: SMTP Authentication Failed. Check SENDER_EMAIL and SENDER_PASSWORD. Details: {e}")
            print("For Gmail, you might need to use an 'App Password' if 2-Factor Authentication is enabled.")
            return False
        except Exception as e:
            print(f"ERROR: Could not send email to {recipient_email}. Details: {e}")
            return False

    def process_frame(self, frame_bytes: bytes):
        """
        Receives a camera frame, processes it for face detection and recognition,
        and updates the robot's state machine based on the workflow diagram.
        Returns a dictionary with robot's message and required client action.
        """
        current_time = time.time()
        
        # Convert bytes to OpenCV image
        try:
            np_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if frame is None:
                print("ERROR: Could not decode image bytes.")
                return self._create_response("Error processing camera feed.", "display_message", self.robot_state)
        except Exception as e:
            print(f"ERROR: Failed to convert frame bytes to image: {e}")
            return self._create_response("Error processing camera feed.", "display_message", self.robot_state)

        detected_bbox, cropped_face_for_embedding = detect_face_in_frame(frame)
        face_detected_this_frame = (detected_bbox is not None)

        # --- Continuous Face Tracking & Loss of Recognition Check (Applies if a person is "active") ---
        # This logic runs continuously to detect if the *currently recognized/interacting* person leaves.
        if self.active_recognized_person:
            if face_detected_this_frame and cropped_face_for_embedding is not None:
                temp_embedding = get_embedding_from_image_or_frame(cropped_face_for_embedding)
                if temp_embedding is not None:
                    best_match_data = find_top_n_neighbors_faiss(
                        temp_embedding,
                        self.faiss_index,
                        self.faiss_index_mapping,
                        self.all_names,
                        CANDIDATE_MATCH_THRESHOLD,
                        n=1
                    )
                    # Check if the best match is still the active person
                    current_best_match_name = best_match_data[0][1] if best_match_data else None
                    
                    if current_best_match_name == self.active_recognized_person:
                        self.loss_of_recognition_counter = 0 # Reset counter if still recognized
                        # Keep extending presence time if needed, but not critical for this flow
                    else:
                        self.loss_of_recognition_counter += 1 # Recognized person's face not matching
                else: # Embedding error for active person's face
                    self.loss_of_recognition_counter += 1
            else: # Active person's face not detected
                self.loss_of_recognition_counter += 1

            if self.loss_of_recognition_counter > LOSS_OF_RECOGNITION_THRESHOLD_FRAMES:
                print(f"[SYSTEM] Lost recognition of {self.active_recognized_person}. Resetting state.")
                # If the active person was being welcomed, send a goodbye message
                if self.robot_state == ROBOT_STATE_WELCOMED:
                    return_msg = self._create_response(f"Goodbye, {self.active_recognized_person}. Please come back soon!", "display_message", ROBOT_STATE_GOODBYE)
                    self._reset_state()
                    return return_msg
                # If they walked away during an interaction (e.g., while waiting for official)
                elif self.robot_state not in [ROBOT_STATE_IDLE, ROBOT_STATE_GOODBYE]: # If not already idle or saying goodbye
                    return_msg = self._create_response(f"It seems you've left. Goodbye.", "display_message", ROBOT_STATE_GOODBYE)
                    self._reset_state()
                    return return_msg

        # --- Main Workflow Logic (Driven by `robot_state`) ---

        if self.robot_state == ROBOT_STATE_IDLE:
            if face_detected_this_frame:
                self.robot_state = ROBOT_STATE_DETECTING_FACE
                self.face_present_start_time = current_time
                print(f"[SYSTEM] Face detected, starting {PRESENCE_DURATION_THRESHOLD_SEC}s timer.")
                return self._create_response("Please face the camera steadily.", "display_message", self.robot_state)
            return self._create_response("Robot is IDLE, waiting for a visitor.", "display_message", self.robot_state)

        elif self.robot_state == ROBOT_STATE_DETECTING_FACE:
            if face_detected_this_frame:
                elapsed_time = current_time - self.face_present_start_time
                if elapsed_time >= PRESENCE_DURATION_THRESHOLD_SEC:
                    self.last_face_image_cv2 = cropped_face_for_embedding # Store for later use
                    if self.last_face_image_cv2 is not None and self.last_face_image_cv2.shape[0] > 0 and self.last_face_image_cv2.shape[1] > 0:
                        self.last_face_embedding = get_embedding_from_image_or_frame(self.last_face_image_cv2)
                        
                        if self.last_face_embedding is not None:
                            self.robot_state = ROBOT_STATE_PROCESSING_RECOGNITION
                            print("[SYSTEM] 3 seconds passed, processing face for recognition.")
                            return self._create_response("Processing your face...", "display_message", self.robot_state)
                        else:
                            print("[SYSTEM] Error generating embedding. Resetting.")
                            return self._reset_state() # Reset on embedding failure
                    else:
                        print("[SYSTEM] Couldn't get a clear image. Resetting.")
                        return self._reset_state() # Reset on bad image
                return self._create_response(f"Face present for {elapsed_time:.1f}s. Hold steady...", "display_message", self.robot_state)
            else:
                print("[SYSTEM] Face lost during detection. Resetting.")
                return self._reset_state() # Reset if face leaves during detection phase

        elif self.robot_state == ROBOT_STATE_PROCESSING_RECOGNITION:
            self.current_identity_candidates = find_top_n_neighbors_faiss(
                self.last_face_embedding,
                self.faiss_index,
                self.faiss_index_mapping,
                self.all_names,
                CANDIDATE_MATCH_THRESHOLD,
                n=MAX_CANDIDATES_TO_PRESENT
            )
            self.current_identity_candidate_index = 0 # Start with the first candidate

            if self.current_identity_candidates:
                # Visitor is potentially known, start confirmation loop
                self.robot_state = ROBOT_STATE_CONFIRMING_IDENTITY
                print("[SYSTEM] Candidates found, moving to identity confirmation.")
                dist, name = self.current_identity_candidates[self.current_identity_candidate_index]
                # "Welcome to XYZ. I believe you are Mr./Ms. ABC. Right?"
                return self._create_response(
                    message=f"Welcome to XYZ. I believe you are Mr./Ms. {name}. Right?",
                    action="request_options_input",
                    state=self.robot_state,
                    options=["yes", "no"]
                )
            else:
                # Visitor is unknown
                self.robot_state = ROBOT_STATE_GREET_UNKNOWN
                print("[SYSTEM] No known candidates, greeting as unknown.")
                # "Welcome to XYZ. Please tell me your name after the beep. Keep face tracking on."
                return self._create_response(
                    message="Welcome to XYZ. Please tell me your name after the beep. Keep face tracking on.",
                    action="request_text_input", # Client should show text input
                    state=self.robot_state,
                    prompt="Your Name" # Prompt for the input field
                )
            
        elif self.robot_state == ROBOT_STATE_WELCOMED:
            # If a visitor is welcomed, we keep tracking them.
            # The continuous face tracking at the top of this function will handle their departure.
            return self._create_response(f"Welcome, {self.active_recognized_person}! Please proceed.", "display_message", self.robot_state)

        elif self.robot_state == ROBOT_STATE_GOODBYE:
            # This state is transient, after a goodbye message, it should reset.
            # This is handled by the loss_of_recognition_counter or explicit reset.
            return self._reset_state()

        # For states that require explicit user input (via /submit_input or /official_response),
        # this `process_frame` function will simply maintain the current message/request
        # until the relevant input API is called.
        return self._create_response("Waiting for your input...", "display_message", self.robot_state)

    def submit_input(self, user_input: str):
        """
        Receives user input (e.g., text, selected option) and processes it
        based on the current robot state, following the workflow diagram.
        Returns a dictionary with robot's message and required client action.
        """
        current_time = time.time() # Re-evaluate time for officer wait

        if self.robot_state == ROBOT_STATE_CONFIRMING_IDENTITY:
            response = user_input.lower().strip()
            if response == 'yes':
                # Visitor confirmed their identity
                dist, name = self.current_identity_candidates[self.current_identity_candidate_index]
                self.active_recognized_person = name # Set the active recognized person
                self.visitor_confirmed_name = name # Store confirmed name
                self.robot_state = ROBOT_STATE_ASK_WHOM_TO_MEET
                print(f"[SYSTEM] Visitor {self.active_recognized_person} confirmed identity. Asking whom to meet.")
                # "Custoimzed message 2, e.g. Whom would you like to meet? Please mention the name after the beep"
                return self._create_response(
                    message=f"Thank you. Welcome, {self.active_recognized_person}. Whom would you like to meet? Please mention the name after the beep.",
                    action="request_text_input",
                    state=self.robot_state,
                    prompt="Official's Name"
                )
            elif response == 'no':
                self.current_identity_candidate_index += 1
                print("[SYSTEM] Visitor denied, checking next candidate.")
                if self.current_identity_candidate_index < len(self.current_identity_candidates):
                    # Ask about the next candidate
                    dist, name = self.current_identity_candidates[self.current_identity_candidate_index]
                    return self._create_response(
                        message=f"Okay. How about Mr./Ms. {name}? Right?",
                        action="request_options_input",
                        state=self.robot_state,
                        options=["yes", "no"]
                    )
                else:
                    # All candidates rejected or no more candidates
                    self.robot_state = ROBOT_STATE_GREET_UNKNOWN
                    print("[SYSTEM] All identity candidates rejected. Treating as unknown.")
                    # "I apologize, I could not confirm your identity. Welcome to XYZ Company. Please tell me your name after the beep."
                    return self._create_response(
                        message="I apologize, I could not confirm your identity. Welcome to XYZ Company. Can you please provide your name after the beep?",
                        action="request_text_input",
                        state=self.robot_state,
                        prompt="Your Name"
                    )
            else:
                # Invalid input for this state
                return self._create_response("I didn't understand. Please say 'Yes' or 'No'.", "request_options_input", self.robot_state, options=["yes", "no"])

        elif self.robot_state == ROBOT_STATE_GREET_UNKNOWN:
            # This state is a prompt, the actual name input is handled in ENROLL_NAME
            # This should ideally not be reached via submit_input directly, but as a fall-through from process_frame
            # If it is reached, it means the client sent input after the initial "Greet Unknown" message
            # We assume the input is the name for enrollment.
            return self._handle_enroll_name_input(user_input)

        elif self.robot_state == ROBOT_STATE_ENROLL_NAME:
            return self._handle_enroll_name_input(user_input)

        elif self.robot_state == ROBOT_STATE_ENROLL_PHOTO_CONFIRM:
            response = user_input.lower().strip()
            if response == 'yes':
                # Visitor confirmed their photo and enrollment
                self.robot_state = ROBOT_STATE_ASK_WHOM_TO_MEET
                print(f"[SYSTEM] Visitor {self.active_recognized_person} enrolled and confirmed photo. Asking whom to meet.")
                # "Thank you. Mr./Ms. [Visitor Name]. Whom would you like to meet? Please mention the name after the beep"
                return self._create_response(
                    message=f"Thank you, {self.active_recognized_person}. Whom would you like to meet? Please mention the name after the beep.",
                    action="request_text_input",
                    state=self.robot_state,
                    prompt="Official's Name"
                )
            elif response == 'no':
                # Visitor denied photo, re-enroll
                print("[SYSTEM] Enrollment photo denied. Removing last entry and resetting.")
                if self.all_face_embeddings and self.all_names and self.all_names[-1] == self.active_recognized_person:
                    self.all_face_embeddings.pop()
                    self.all_names.pop()
                    # Rebuild FAISS index after removal
                    self.faiss_index = None
                    self.faiss_index_mapping = []
                    if self.all_face_embeddings:
                        embeddings_for_faiss = np.array(self.all_face_embeddings, dtype=np.float32)
                        D = embeddings_for_faiss.shape[1]
                        faiss_index_temp = faiss.IndexFlatL2(D) # Create a new index
                        faiss_index_temp.add(embeddings_for_faiss)
                        self.faiss_index = faiss_index_temp # Assign the new index
                        self.faiss_index_mapping = list(range(len(self.all_names)))
                    save_known_faces(self.all_face_embeddings, self.all_names, self.face_data_path, self.names_path)
                return self._reset_state() # Go back to IDLE, will re-detect and re-enroll
            else:
                return self._create_response("I didn't understand. Please say 'Yes' or 'No'.", "request_options_input", self.robot_state, options=["yes", "no"])

        elif self.robot_state == ROBOT_STATE_ASK_WHOM_TO_MEET:
            self.whom_to_meet_spoken_name_raw = user_input.strip()
            if not self.whom_to_meet_spoken_name_raw:
                return self._create_response("I didn't get the name. Please speak clearly.", "request_text_input", self.robot_state, prompt="Official's Name")
            
            # Find closest matching official names
            self.whom_to_meet_candidates = self._find_matching_officials(self.whom_to_meet_spoken_name_raw)
            self.whom_to_meet_candidate_index = 0

            if self.whom_to_meet_candidates:
                self.robot_state = ROBOT_STATE_CONFIRM_WHO_TO_MEET
                # "A screen appears with the names most likely to what the visitor has spoken on the top of the list from which the visitor selects the desired one"
                # For now, we'll ask for confirmation of the top one or present options.
                top_candidate_name = self.whom_to_meet_candidates[0]
                return self._create_response(
                    message=f"Did you mean {top_candidate_name}? Or choose from the list.",
                    action="request_options_input",
                    state=self.robot_state,
                    options=[top_candidate_name] + [name for name in self.whom_to_meet_candidates[1:]] + ["none of these"]
                )
            else:
                # No matching official found
                return self._create_response(
                    message="I couldn't find an official by that name. Please try again or provide a different name.",
                    action="request_text_input",
                    state=self.robot_state,
                    prompt="Official's Name"
                )

        elif self.robot_state == ROBOT_STATE_CONFIRM_WHO_TO_MEET:
            selected_official = user_input.strip().lower()
            if selected_official in [name.lower() for name in self.whom_to_meet_candidates]:
                self.whom_to_meet_confirmed_name = selected_official.title() # Store confirmed official name
                self.robot_state = ROBOT_STATE_CONTACTING_OFFICIAL
                print(f"[SYSTEM] Visitor {self.active_recognized_person} wants to meet {self.whom_to_meet_confirmed_name}.")
                # "Customized message e.g. Thanks. Please wait while I inform Mr./Ms. MNO about you"
                return self._create_response(
                    message=f"Thanks. Please wait while I inform Mr./Ms. {self.whom_to_meet_confirmed_name} about you.",
                    action="display_message",
                    state=self.robot_state
                )
            elif selected_official == "none of these":
                return self._create_response(
                    message="Okay, please try again. Whom would you like to meet? Please mention the name after the beep.",
                    action="request_text_input",
                    state=self.robot_state,
                    prompt="Official's Name"
                )
            else:
                return self._create_response("I didn't understand. Please select from the options.", "request_options_input", self.robot_state, options=[name for name in self.whom_to_meet_candidates] + ["none of these"])

        elif self.robot_state == ROBOT_STATE_CONTACTING_OFFICIAL:
            # This state is primarily for internal action, not direct user input.
            # It transitions to WAITING_FOR_OFFICIAL_RESPONSE immediately after sending notification.
            if self._send_notification_to_official(
                official_name=self.whom_to_meet_confirmed_name,
                visitor_name=self.active_recognized_person,
                photo_cv2_frame=self.last_face_image_cv2,
                message_type="visitor_arrival"
            ):
                self.official_contact_start_time = current_time
                self.official_contact_iteration = 1
                self.robot_state = ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE
                return self._create_response(
                    message=f"I have informed Mr./Ms. {self.whom_to_meet_confirmed_name}. Please wait a moment for their response.",
                    action="display_message",
                    state=self.robot_state
                )
            else:
                # Failed to send notification (e.g., official not in system)
                return self._create_response(
                    message=f"I'm sorry, I couldn't find contact information for {self.whom_to_meet_confirmed_name}. Please try again with a different name.",
                    action="request_text_input",
                    state=ROBOT_STATE_ASK_WHOM_TO_MEET,
                    prompt="Official's Name"
                )

        elif self.robot_state == ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS:
            response = user_input.lower().strip()
            if response == 'wait':
                self.official_contact_start_time = current_time # Reset timer for next wait
                self.robot_state = ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE
                print("[SYSTEM] Visitor opted to wait. Re-entering waiting state.")
                return self._create_response(
                    message=f"Okay, I will keep trying to contact Mr./Ms. {self.whom_to_meet_confirmed_name}. Please wait.",
                    action="display_message",
                    state=self.robot_state
                )
            elif response == 'leave message':
                self.robot_state = ROBOT_STATE_LEAVE_MESSAGE
                print("[SYSTEM] Visitor opted to leave a message.")
                return self._create_response(
                    message="Please speak your message after the beep.",
                    action="request_text_input",
                    state=self.robot_state,
                    prompt="Your Message"
                )
            elif response == 'leave':
                print("[SYSTEM] Visitor opted to leave.")
                return self._reset_state() # Resets all state, effectively ending interaction
            else:
                return self._create_response(
                    message="I didn't understand. Please say 'Wait', 'Leave Message', or 'Leave'.",
                    action="request_options_input",
                    state=self.robot_state,
                    options=["wait", "leave message", "leave"]
                )
        
        elif self.robot_state == ROBOT_STATE_LEAVE_MESSAGE:
            visitor_message = user_input.strip()
            if visitor_message:
                print(f"[SYSTEM] Visitor left message: '{visitor_message}' for {self.whom_to_meet_confirmed_name}.")
                if self._send_notification_to_official(
                    official_name=self.whom_to_meet_confirmed_name,
                    visitor_name=visitor_message, # This will be the message content
                    message_type="visitor_message"
                ):
                    return_msg = self._create_response("Your message has been sent. Thank you.", "display_message", ROBOT_STATE_GOODBYE)
                    self._reset_state() # End interaction after message sent
                    return return_msg
                else:
                    return_msg = self._create_response("I'm sorry, I encountered an issue sending your message.", "display_message", ROBOT_STATE_GOODBYE)
                    self._reset_state()
                    return return_msg
            else:
                return self._create_response("I didn't get your message. Please speak clearly.", "request_text_input", self.robot_state, prompt="Your Message")

        # Fallback for unexpected input in other states
        return self._create_response("Unexpected input. Robot is in " + self.robot_state + " state.", "display_message", self.robot_state)

    def _handle_enroll_name_input(self, user_input: str):
        """Helper function to handle name input for unknown visitors."""
        visitor_name = user_input.strip()
        if not visitor_name:
            # "Sorry for not being able to recognize you. Can you please provide your name after the beep?"
            return self._create_response("I didn't get your name. Please speak clearly.", "request_text_input", ROBOT_STATE_ENROLL_NAME, prompt="Your Name")
        
        self.visitor_spoken_name_raw = visitor_name # Store raw spoken name
        
        # Simulate name selection/editing (for now, just confirm the spoken name)
        # In a real system, you'd perform fuzzy matching against a directory or suggest corrections.
        # For this flow, we'll ask for confirmation of the provided name.
        self.active_recognized_person = visitor_name # Temporarily set active person to the spoken name
        
        # Add to known faces immediately for subsequent recognition, then confirm
        # In a real system, you might wait for confirmation before adding permanently.
        if self.last_face_embedding is not None:
            self.all_face_embeddings.append(self.last_face_embedding.tolist())
            self.all_names.append(visitor_name)
            
            # Rebuild FAISS index with new data
            self.faiss_index = None # Clear old index
            self.faiss_index_mapping = []
            if self.all_face_embeddings:
                embeddings_for_faiss = np.array(self.all_face_embeddings, dtype=np.float32)
                D = embeddings_for_faiss.shape[1]
                faiss_index_temp = faiss.IndexFlatL2(D) # Create a new index
                faiss_index_temp.add(embeddings_for_faiss)
                self.faiss_index = faiss_index_temp # Assign the new index
                self.faiss_index_mapping = list(range(len(self.all_names)))
            save_known_faces(self.all_face_embeddings, self.all_names, self.face_data_path, self.names_path)
            print(f"[SYSTEM] Visitor '{visitor_name}' temporarily added to known faces for session.")
        else:
            print("[SYSTEM] No face embedding available for enrollment. Cannot enroll.")
            return self._create_response("I couldn't get your photo for enrollment. Please try again.", "display_message", ROBOT_STATE_IDLE)


        self.robot_state = ROBOT_STATE_ENROLL_PHOTO_CONFIRM # Transition to photo confirmation
        # "A screen appears with the name as heard by the robot on the screen which can be edited by the visitor on the screen. Also, best picture from recently take photo will be there as well"
        # Since we don't have a direct screen edit, we ask for confirmation of the name and photo.
        
        # Convert last_face_image_cv2 to base64 for display on client
        image_base64 = None
        if self.last_face_image_cv2 is not None:
            _, img_encoded = cv2.imencode('.jpg', self.last_face_image_cv2)
            image_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return self._create_response(
            message=f"Thank you. Is this your name: {visitor_name}? And is this your photo?",
            action="request_options_input", # Client should show options for confirmation
            state=self.robot_state,
            options=["yes", "no"],
            image_base64=image_base64 # Include the image for display
        )

    def _find_matching_officials(self, spoken_name: str, n: int = MAX_CANDIDATES_TO_PRESENT):
        """
        Simulates finding matching company officials based on spoken name.
        In a real system, this would use fuzzy string matching or a more sophisticated search.
        For now, it's a simple case-insensitive check and substring match.
        """
        spoken_name_lower = spoken_name.lower()
        matches = []
        for official_name in self.company_officials.keys():
            if spoken_name_lower in official_name: # Simple substring match
                matches.append(official_name.title()) # Return in title case
        
        # Sort by exact match first, then by length or alphabetical
        matches.sort(key=lambda x: (x.lower() != spoken_name_lower, len(x), x))
        return matches[:n] # Return top N matches

    def check_official_timeout(self):
        """
        Checks if the official response has timed out and updates state accordingly.
        This method will be called periodically by the FastAPI application.
        """
        if self.robot_state == ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE and self.official_contact_start_time:
            elapsed_time = time.time() - self.official_contact_start_time
            if elapsed_time >= OFFICIAL_RESPONSE_TIMEOUT_SEC:
                self.official_contact_iteration += 1
                if self.official_contact_iteration <= MAX_OFFICIAL_CONTACT_ITERATIONS:
                    print(f"[SYSTEM] Official response timed out (Iteration {self.official_contact_iteration}). Re-contacting.")
                    # Re-send notification
                    if self._send_notification_to_official(
                        official_name=self.whom_to_meet_confirmed_name,
                        visitor_name=self.active_recognized_person,
                        photo_cv2_frame=self.last_face_image_cv2,
                        message_type="visitor_arrival" # Or a "reminder" type
                    ):
                        self.official_contact_start_time = time.time() # Reset timer for next wait
                        return self._create_response(
                            message=f"Still waiting for Mr./Ms. {self.whom_to_meet_confirmed_name}'s response. Retrying contact... ({self.official_contact_iteration}/{MAX_OFFICIAL_CONTACT_ITERATIONS})",
                            action="display_message",
                            state=self.robot_state
                        )
                    else:
                        # Fallback if re-contact fails
                        print("[SYSTEM] Re-contact failed. Transitioning to unreachable options.")
                        self.robot_state = ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS
                        return self._create_response(
                            message=f"I am afraid Mr./Ms. {self.whom_to_meet_confirmed_name} is unreachable. You may wait and I will keep trying or you may leave a message.",
                            action="request_options_input",
                            state=self.robot_state,
                            options=["wait", "leave message", "leave"]
                        )
                else:
                    print("[SYSTEM] Max official contact iterations reached. Official unreachable.")
                    self.robot_state = ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS
                    # "Customised Message while approaching/calling visitor e.g. Hey, I am afraid Mr./Ms. MNO is unreachable. You may wait and I will keep trying or you may leave a message"
                    return self._create_response(
                        message=f"I am afraid Mr./Ms. {self.whom_to_meet_confirmed_name} is unreachable. You may wait and I will keep trying or you may leave a message.",
                        action="request_options_input",
                        state=self.robot_state,
                        options=["wait", "leave message", "leave"]
                    )
        # Return current state message if no timeout or not in waiting state
        return self._create_response("No timeout action needed.", "no_action", self.robot_state)

    def receive_official_response(self, response_text: str):
        """
        Receives a response from the company official (simulated via API).
        This will typically be "approve" or "deny".
        """
        response_text_lower = response_text.lower().strip()
        if self.robot_state == ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE or \
           self.robot_state == ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS: # Can approve/deny even if unreachable
            
            if "approve" in response_text_lower:
                self.robot_state = ROBOT_STATE_WELCOMED
                print(f"[SYSTEM] Official approved visitor {self.active_recognized_person}.")
                # "Customized Message based upon the response from the app on company official's response"
                return self._create_response(
                    message=f"Great news! Mr./Ms. {self.whom_to_meet_confirmed_name} has approved your visit. Please proceed.",
                    action="display_message",
                    state=self.robot_state
                )
            elif "deny" in response_text_lower:
                print(f"[SYSTEM] Official denied visitor {self.active_recognized_person}. Resetting state.")
                return self._reset_state() # Denied, reset and end interaction
            else:
                return self._create_response(
                    message="Official response not understood. Waiting for 'approve' or 'deny'.",
                    action="display_message", # Keep displaying current state message
                    state=self.robot_state
                )
        else:
            print(f"ERROR: Received official response '{response_text}' in unexpected state: {self.robot_state}.")
            return self._create_response("Received unexpected official response.", "display_message", self.robot_state)