# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
import io
import base64
import asyncio
import time # Import time for elapsed calculation in /status
from dotenv import load_dotenv # For loading environment variables from .env file
load_dotenv()
# Import the RobotBrain class and state definitions
from robot_brain import (
    RobotBrain,
    ROBOT_STATE_IDLE,
    ROBOT_STATE_DETECTING_FACE,
    ROBOT_STATE_PROCESSING_RECOGNITION,
    ROBOT_STATE_CONFIRMING_IDENTITY,
    ROBOT_STATE_GREET_UNKNOWN,
    ROBOT_STATE_ENROLL_NAME,
    ROBOT_STATE_ENROLL_PHOTO_CONFIRM,
    ROBOT_STATE_ASK_WHOM_TO_MEET,
    ROBOT_STATE_CONFIRM_WHO_TO_MEET,
    ROBOT_STATE_CONTACTING_OFFICIAL,
    ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE,
    ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS,
    ROBOT_STATE_LEAVE_MESSAGE,
    ROBOT_STATE_WELCOMED,
    ROBOT_STATE_GOODBYE
)

# --- Configuration ---
# IMPORTANT: Adjust this BASE_PROJECT_FOLDER to your actual project path
# This path should contain your model, faces_data.pkl, names.pkl
BASE_PROJECT_FOLDER = r"E:\Buffer\Applied data science and AI specialization\NCL open cv tasks\app3"
MODEL_PATH = os.path.join(BASE_PROJECT_FOLDER, "Resnet50_3.pth")
FACE_DATA_PKL = os.path.join(BASE_PROJECT_FOLDER, "faces_data.pkl")
NAMES_PKL = os.path.join(BASE_PROJECT_FOLDER, "names.pkl")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Receptionist Robot Backend",
    description="NONE.",
    version="1.0.0",
)

# Instantiate the RobotBrain (this will load models and known faces once at startup)
# This instance will persist throughout the application's lifetime
robot_brain = RobotBrain(
    base_project_folder=BASE_PROJECT_FOLDER,
    model_path=MODEL_PATH,
    face_data_path=FACE_DATA_PKL,
    names_path=NAMES_PKL
)

# --- Pydantic Models for API Request/Response Bodies ---

class RobotResponse(BaseModel):
    """Defines the structure of the robot's response to the client."""
    message: str
    action: str  # e.g., "display_message", "request_text_input", "request_options_input", "no_action"
    state: str   # Current state of the robot
    options: list[str] | None = None  # For "request_options_input" action
    prompt: str | None = None # For "request_text_input" action
    image_base64: str | None = None # For displaying captured photo (e.g., during enrollment confirmation)

class UserInput(BaseModel):
    """Defines the structure for user input from the client."""
    input_text: str

class OfficialMessage(BaseModel):
    """Defines the structure for a message from an official (simulated external system)."""
    message: str

# --- Background Task for Official Timeout Check ---
async def check_official_timeout_periodically():
    """
    Periodically checks for official response timeout and updates robot state.
    Runs as a background task.
    """
    while True:
        # Only check if robot is in a waiting state
        if robot_brain.robot_state == ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE or \
           robot_brain.robot_state == ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS:
            
            response = robot_brain.check_official_timeout()
            # If check_official_timeout returned a state change/message, log it.
            # No need to send back to client directly from here, as client polls via process_frame/status.
            if response and response.get("action") != "no_action":
                print(f"[BACKGROUND TASK] Official timeout check triggered: {response.get('message')}")
        
        await asyncio.sleep(10) # Check every 10 seconds

@app.on_event("startup")
async def startup_event():
    """Starts the background task for official timeout check on application startup."""
    asyncio.create_task(check_official_timeout_periodically())
    print("FastAPI application started. Background task for official timeout initiated.")


# --- API Endpoints ---

@app.get("/", response_model=RobotResponse)
async def root():
    """
    Root endpoint to get initial robot status.
    """
    return RobotResponse(
        message="Receptionist Robot Backend is running. Waiting for camera feed.",
        action="display_message",
        state=robot_brain.robot_state
    )

@app.post("/process_frame", response_model=RobotResponse)
async def process_camera_frame(file: UploadFile = File(...)):
    """
    Receives a camera frame (image file) from the client, processes it for face detection,
    recognition, and updates the robot's state machine based on the workflow diagram.
    Returns the robot's current message and required client action.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

    try:
        frame_bytes = await file.read()
        
        # Process the frame using the RobotBrain
        response = robot_brain.process_frame(frame_bytes)
        
        return RobotResponse(**response)
    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during frame processing: {e}")

@app.post("/submit_input", response_model=RobotResponse)
async def submit_user_input(user_input: UserInput):
    """
    Receives user input (e.g., text, selected option) from the client
    and processes it based on the robot's current state, following the workflow diagram.
    """
    try:
        response = robot_brain.submit_input(user_input.input_text)
        return RobotResponse(**response)
    except Exception as e:
        print(f"Error submitting input: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during input submission: {e}")

@app.post("/official_response", response_model=RobotResponse)
async def receive_official_response(official_msg: OfficialMessage):
    """
    Simulates a company official sending a response (e.g., "approve", "deny")
    to the robot, influencing its state.
    """
    try:
        response = robot_brain.receive_official_response(official_msg.message)
        return RobotResponse(**response)
    except Exception as e:
        print(f"Error handling official message: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error handling official message: {e}")

@app.get("/status", response_model=RobotResponse)
async def get_robot_status():
    """
    Returns the current status and message of the robot.
    Useful for clients to poll for updates or for debugging.
    """
    # This endpoint provides a way to get the robot's current state without sending a frame or input.
    # It will return the last message and action based on the robot's internal state.
    # For a more robust solution, the RobotBrain might store the *last generated response*
    # which this endpoint could then return. For simplicity, we regenerate a default.
    
    current_state = robot_brain.robot_state
    message = "Robot is in an unknown state."
    action = "display_message"
    options = None
    prompt = None
    image_base64 = None

    if current_state == ROBOT_STATE_IDLE:
        message = "Robot is IDLE, waiting for a visitor."
    elif current_state == ROBOT_STATE_DETECTING_FACE:
        message = "Visitor detected, timing for 3 seconds..."
    elif current_state == ROBOT_STATE_PROCESSING_RECOGNITION:
        message = "Processing face for recognition..."
    elif current_state == ROBOT_STATE_CONFIRMING_IDENTITY:
        # If in this state, it means it's waiting for "yes/no" on identity
        message = f"I believe you are Mr./Ms. {robot_brain.current_identity_candidates[robot_brain.current_identity_candidate_index][1] if robot_brain.current_identity_candidates else 'someone'}. Right?"
        action = "request_options_input"
        options = ["yes", "no"]
    elif current_state == ROBOT_STATE_GREET_UNKNOWN:
        message = "Welcome to XYZ. Please tell me your name after the beep. Keep face tracking on."
        action = "request_text_input"
        prompt = "Your Name"
    elif current_state == ROBOT_STATE_ENROLL_NAME:
        message = "Please tell me your full name now."
        action = "request_text_input"
        prompt = "Your Name"
    elif current_state == ROBOT_STATE_ENROLL_PHOTO_CONFIRM:
        message = f"Thank you. Is this your name: {robot_brain.active_recognized_person}? And is this your photo?"
        action = "request_options_input"
        options = ["yes", "no"]
        # Include the image if available
        if robot_brain.last_face_image_cv2 is not None:
            _, img_encoded = cv2.imencode('.jpg', robot_brain.last_face_image_cv2)
            image_base64 = base64.b64encode(img_encoded).decode('utf-8')
    elif current_state == ROBOT_STATE_ASK_WHOM_TO_MEET:
        message = "Whom would you like to meet? Please mention the name after the beep."
        action = "request_text_input"
        prompt = "Official's Name"
    elif current_state == ROBOT_STATE_CONFIRM_WHO_TO_MEET:
        message = f"Did you mean {robot_brain.whom_to_meet_candidates[0] if robot_brain.whom_to_meet_candidates else 'this person'}? Or choose from the list."
        action = "request_options_input"
        options = [name for name in robot_brain.whom_to_meet_candidates] + ["none of these"]
    elif current_state == ROBOT_STATE_CONTACTING_OFFICIAL:
        message = f"Thanks. Please wait while I inform Mr./Ms. {robot_brain.whom_to_meet_confirmed_name} about you."
    elif current_state == ROBOT_STATE_WAITING_FOR_OFFICIAL_RESPONSE:
        elapsed = int(time.time() - robot_brain.official_contact_start_time) if robot_brain.official_contact_start_time else 0
        message = f"Waiting for official response from Mr./Ms. {robot_brain.whom_to_meet_confirmed_name}... ({elapsed}s)"
        action = "display_message" # Client should just display this, official response comes via /official_response
    elif current_state == ROBOT_STATE_OFFICIAL_UNREACHABLE_OPTIONS:
        message = f"I am afraid Mr./Ms. {robot_brain.whom_to_meet_confirmed_name} is unreachable. You may wait and I will keep trying or you may leave a message."
        action = "request_options_input"
        options = ["wait", "leave message", "leave"]
    elif current_state == ROBOT_STATE_LEAVE_MESSAGE:
        message = "Please speak your message after the beep."
        action = "request_text_input"
        prompt = "Your Message"
    elif current_state == ROBOT_STATE_WELCOMED:
        message = f"Welcome, {robot_brain.active_recognized_person}! Please proceed."
    elif current_state == ROBOT_STATE_GOODBYE:
        message = "Goodbye. Thank you for your visit."
        action = "display_message" # This state immediately resets to IDLE

    return RobotResponse(
        message=message,
        action=action,
        state=current_state,
        options=options,
        prompt=prompt,
        image_base64=image_base64
    )


# --- How to Run the FastAPI Application ---
# To run this FastAPI application, you'll need uvicorn.
# 1. Ensure you have `uvicorn`, `fastapi`, `opencv-python`, `mediapipe`, `Pillow`, `torch`, `torchvision`, `scipy`, `faiss-cpu`, and `pyttsx3` installed:
#    pip install uvicorn fastapi opencv-python mediapipe Pillow torch torchvision scipy faiss-cpu pyttsx3
# 2. Save the above code as `main.py` in the same directory as `utils.py` and `robot_brain.py`.
# 3. Open your terminal in that directory.
# 4. Run the command:
#    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#    - `main`: refers to the `main.py` file.
#    - `app`: refers to the `FastAPI()` instance named `app` inside `main.py`.
#    - `--host 0.0.0.0`: makes the server accessible from other devices on your network (useful for mobile app).
#    - `--port 8000`: runs the server on port 8000.
#    - `--reload`: automatically reloads the server on code changes (useful during development).
#
# Once running, you can access the interactive API documentation at:
# http://127.0.0.1:8000/docs (Swagger UI)
# http://127.0.0.1:8000/redoc (ReDoc)
#
# Your mobile app would then send POST requests to http://<your_backend_ip>:8000/process_frame
# and http://<your_backend_ip>:8000/submit_input.
