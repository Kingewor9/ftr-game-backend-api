import hmac
import hashlib
import json
from urllib.parse import unquote_plus
from fastapi import FastAPI, Request, HTTPException, Body
from pydantic import BaseModel, Field
from urllib.parse import unquote_plus, parse_qsl # <-- CRITICAL NEW IMPORT
from datetime import datetime, timedelta
from fastapi import Depends
import os
import random
import logging
import string
# NEW IMPORT REQUIRED FOR CORS FIX
from fastapi.middleware.cors import CORSMiddleware 
# >>> REQUIRED IMPORT FOR SERVING HTML FRONTEND <<<
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List # Required for Python 3.9+ type hinting
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.client import Client as FirestoreClient # For type hinting
from google.cloud.firestore_v1.base_document import DocumentReference, DocumentSnapshot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# CRITICAL: LOADED FROM ENVIRONMENT VARIABLE
# Make absolutely sure TELEGRAM_BOT_TOKEN is set in your environment
BOT_TOKEN: Optional [str] = os.getenv ("BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("[FATAL ERROR] Cannot proceed: BOT_TOKEN environment variable is NOT set. Hash validation will fail.")
else:
    logger.info(f"[DIAGNOSTIC] BOT_TOKEN successfully loaded. Length: {len(BOT_TOKEN)}.")

# Define the admin's Telegram ID for exclusive access to admin endpoints
ADMIN_TELEGRAM_ID = "1474715816"

# 1. Retrieve the raw JSON string from the environment variable
# We use .get() for safety and provide a default empty string if the variable isn't set.
raw_firebase_config = os.environ.get('__FIREBASE_CONFIG', '')

# 2. Parse the JSON string into a Python dictionary.
# If the string is empty or parsing fails, we default to an empty dictionary.
try:
    if raw_firebase_config:
        firebaseConfig = json.loads(raw_firebase_config)
    else:
        firebaseConfig = {}
        logging.warning("Environment variable '__FIREBASE_CONFIG' not found or is empty.")
except json.JSONDecodeError as e:
    logging.error(f"Failed to decode Firebase config JSON: {e}")
    firebaseConfig = {}


# 3. Retrieve the Application ID from the environment variable.
# Default to 'default-app-id' if not set.
appId = os.environ.get('__APP_ID', 'default-app-id')


# --- Verification (for your own testing) ---

logging.info(f"Loaded App ID: {appId}")
logging.info(f"Firebase Config Keys: {list(firebaseConfig.keys()) if firebaseConfig else 'No Config'}")

# You can now use firebaseConfig and appId in your backend logic
# Example: print(firebaseConfig.get('projectId', 'Project ID not found'))

# A flag to ensure Firebase is initialized only once
_firebase_initialized = False

def initialize_firebase():
    """Initializes Firebase Admin SDK."""
    global _firebase_initialized
    if not _firebase_initialized and firebaseConfig:
        try:
            # We use a placeholder credentials object; the environment handles auth
            cred = credentials.Certificate(firebaseConfig)
            firebase_admin.initialize_app(cred, firebaseConfig)
            _firebase_initialized = True
            print("Firebase Admin SDK initialized successfully.")
        except Exception as e:
            # This can fail in local dev without proper credentials
            print(f"Error initializing Firebase Admin SDK: {e}")

def get_database() -> FirestoreClient:
    """Dependency for getting the Firestore client."""
    if not _firebase_initialized:
        initialize_firebase()
    
    return firestore.client()

# --- FIREBASE PATH CONSTANTS ---

# Leagues are public data that all users of the app can access by code
LEAGUES_COLLECTION_PATH = f"artifacts/{appId}/public/data/leagues"
# User profiles are stored privately by user ID
USERS_PROFILE_COLLECTION_PATH = f"artifacts/{appId}/users"


# --- Pydantic Data Models ---

# Quiz Models (UNCHANGED)
class QuizQuestion(BaseModel):
    q_id: int
    question: str
    options: list[str]
    correct_option_index: int
    
class DailyQuizData(BaseModel):
    quiz_name: str
    difficulty: str
    description: str
    time_limit_seconds: int
    points_per_question: int
    expiration_minutes: int = 1440
    questions: list[QuizQuestion] # <--- THIS LINE IS CRUCIAL

class AnswerSubmission(BaseModel):
    telegram_id: str = Field(...)
    quiz_id: int = Field(...)
    question_id: int = Field(...)
    selected_option_index: int = Field(..., description="Index of the option chosen (0-3).")

# Common Models (UNCHANGED)
class NotificationPreferences(BaseModel):
    daily_reminders: bool = Field(default=True)
    league_updates: bool = Field(default=True)
    achievement_notifications: bool = Field(default=True)

class UserProfileEdit(BaseModel):
    telegram_id: str = Field(...)
    username: str
    email: str | None = None
    avatar_url: str | None = None

class UserPreferencesUpdate(BaseModel):
    telegram_id: str = Field(...)
    preferences: NotificationPreferences
    
class TelegramID(BaseModel):
    telegram_id: str = Field(..., description="The unique ID of the Telegram user.")
    
# --- NEW LEAGUE MODELS ---

class LeagueCreation(BaseModel):
    telegram_id: str = Field(..., description="The creator's Telegram ID.")
    name: str = Field(..., max_length=50)
    description: str = Field(..., max_length=200)
    is_private: bool = Field(default=False, description="True for private leagues (uses code), False for public.")
    member_limit: int = Field(default=50, ge=2, le=500)
    start_date: str = Field(default=datetime.now().strftime("%Y-%m-%d"), description="League start date (YYYY-MM-DD)")
    league_avatar_url: str | None = None
    difficulty_tag: str = Field(default="Medium", description="e.g., Easy, Medium, Hard")

class LeagueJoin(BaseModel):
    telegram_id: str = Field(..., description="The joining user's Telegram ID.")
    code: str = Field(..., description="The 6-digit join code for private leagues.")

class LeagueSearch(BaseModel):
    telegram_id: str = Field(..., description="The searching user's Telegram ID.")
    query: str = Field(..., description="Search term for public leagues.")
    
    # NEW MODEL: Now uses 'league_id' to match the field name sent by the frontend
class LeagueDetailsRequest(BaseModel):
    telegram_id: str = Field(..., description="The user requesting the leaderboard.")
    league_id: str = Field(..., description="The 6-digit code/ID of the league to view.")

# --- Global In-Memory State (Used only for Daily Quiz, NOT for persistent user/league data) ---
# NOTE: user_db and league_db global dictionaries have been REMOVED as they are now replaced by Firestore.

daily_quiz_state = {
    "quiz_id": 0,
    "quiz_data": None,
    "start_time": None,
    "expiration_time": None
}
user_quiz_progress = {} 


# --- Helper Functions ---


# !!! CORRECTED AUTHENTICATION LOGIC !!!
def validate_telegram_data(init_data: str) -> dict:
    """ Validates the hash of the received Telegram Mini App init data. (UNCHANGED) """
    # ... (Authentication Logic remains the same)
    print(f"\n[DEBUG] Raw init_data received: {init_data}")
    
    if not BOT_TOKEN:
        print("[FATAL ERROR] Cannot validate hash: BOT_TOKEN is missing (Value is None).")
        raise HTTPException(status_code=500, detail="Server misconfiguration: Telegram Bot Token is missing.")

    params = parse_qsl(init_data, keep_blank_values=True, encoding='utf-8')
    
    hash_value = ""
    data_check_string_parts = []
    user_data_str = ""

    for key, value in params:
        if key == 'hash':
            hash_value = value
        else:
            data_check_string_parts.append(f"{key}={value}")
            if key == 'user':
                user_data_str = value

    if not hash_value or not user_data_str:
        print("[CRITICAL ERROR] Hash or User data was not found.")
        raise HTTPException(status_code=400, detail="Missing required Telegram data.")
        
    data_check_string_parts.sort()
    data_check_string = "\n".join(data_check_string_parts)
    
    secret_key = hmac.new(
        key="WebAppData".encode('utf8'), 
        msg=BOT_TOKEN.encode('utf8'), 
        digestmod=hashlib.sha256
    ).digest()

    calculated_hash = hmac.new(
        secret_key,
        msg=data_check_string.encode('utf8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    if calculated_hash != hash_value:
        print(f"[ERROR] Hash Mismatch Detected! Calculated: {calculated_hash}, Received: {hash_value}")
        raise HTTPException(status_code=403, detail="Invalid Telegram data hash.")
    
    print(f"[SUCCESS] Hash validated successfully: {calculated_hash}")

    try:
        user_info = json.loads(unquote_plus(user_data_str))
        return user_info
    except json.JSONDecodeError as e:
        print(f"JSON decode error in user data: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in user data.")


def calculate_accuracy(correct, total):
    return round((correct / total) * 100) if total > 0 else 0

def generate_league_code(db: FirestoreClient, length=6) -> str:
    """Generates a unique 6-digit alphanumeric code by checking Firestore."""
    chars = string.ascii_uppercase + string.digits
    collection_ref = db.collection(LEAGUES_COLLECTION_PATH)
    
    for _ in range(10): # Try up to 10 times to find a unique code
        code = ''.join(random.choice(chars) for _ in range(length))
        
        # Check if the document (league) exists in the persistent store
        doc = collection_ref.document(code).get()
        if not doc.exists:
            return code
            
    raise HTTPException(status_code=500, detail="Could not generate a unique league code. Please try again.")


# --- API Setup ---
app = FastAPI()


# =======================================================================
# >>> CRITICAL CORS FIX INSERTED HERE <<<
# =======================================================================


# 1. DEFINE YOUR FRONTEND ORIGINS
# When deploying a Telegram Mini App on a phone, the origin (where the request comes from) 
# is often an internal Telegram domain or can be masked. 
# Allowing '*' is necessary for seamless operation in the Mini App environment.
origins = [
    "*", # Allow all origins for seamless Telegram Mini App deployment
]

# Note: If deploying outside a test environment, you should tighten this to specific Telegram domains 
# (e.g., 'https://web.telegram.org') or your specific frontend URL if possible.

# 2. ADD THE MIDDLEWARE TO THE APP
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (POST, GET, PUT, etc.)
    allow_headers=["*"], # Allow all headers 
)

# =======================================================================
# >>> NEW: HEALTH CHECK ENDPOINT <<<
# Use this to confirm the API is running and not being shadowed by the mount
# =======================================================================
@app.get("/api/status")
async def health_check():
    """A simple endpoint to verify the API server is alive."""
    return {"status": "ok", "message": "API is running and accessible."}

# =======================================================================

# --- CORE API Endpoints (Login and Profile - NOW PERSISTENT) ---


@app.post("/auth/login")
async def telegram_login(request: Request, db: FirestoreClient = Depends(get_database)):
    """Authenticates the user and retrieves/creates a persistent profile."""
    try:
        body = await request.body()
        init_data = body.decode('utf-8')
        
        user_info = validate_telegram_data(init_data)
        telegram_id = str(user_info.get("id"))
        
        user_ref: DocumentReference = db.collection(USERS_PROFILE_COLLECTION_PATH).document(telegram_id)
        user_doc: DocumentSnapshot = user_ref.get()
        
        user_profile: Dict[str, Any]
        
        if not user_doc.exists:
            # CREATE NEW USER PROFILE (Persistent)
            username = user_info.get("username") or user_info.get("first_name")
            new_user = {
                "telegram_id": telegram_id,
                "username": username,
                "email": None,
                "avatar_url": None,
                "member_since": datetime.now().strftime("%B %Y"),
                "overall_score": 0,
                "total_quizzes_answered": 0,
                "correct_answers": 0,
                "accuracy_rate": 0,
                "current_streak": 0,
                "best_streak": 0,
                "leagues": {}, # {code: league_points}
                "past_accuracy": [0, 0, 0], 
                "preferences": NotificationPreferences().dict(),
                "completed_quizzes": [] 
            }
            user_ref.set(new_user)
            user_profile = new_user
            print(f"New persistent user created: {username} (ID: {telegram_id})")
        else:
            # RETRIEVE EXISTING USER PROFILE (Persistent)
            user_profile = user_doc.to_dict()
            
        return {
            "status": "success",
            "message": "User authenticated and persistent profile retrieved.",
            "user_profile": user_profile,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An error occurred during login: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.put("/profile/edit")
async def edit_profile(profile_data: UserProfileEdit, db: FirestoreClient = Depends(get_database)):
    """Updates user profile fields and saves to Firestore."""
    user_id = profile_data.telegram_id
    
    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_ref.update({
        "username": profile_data.username,
        "email": profile_data.email,
        "avatar_url": profile_data.avatar_url
    })
    
    updated_profile = user_ref.get().to_dict()
    
    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "user_profile": updated_profile
    }

@app.put("/profile/preferences")
async def update_preferences(prefs_data: UserPreferencesUpdate, db: FirestoreClient = Depends(get_database)):
    """Updates user notification preferences and saves to Firestore."""
    user_id = prefs_data.telegram_id
    
    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(user_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_ref.update({
        "preferences": prefs_data.preferences.dict()
    })
    
    updated_profile = user_ref.get().to_dict()

    return {
        "status": "success",
        "message": "Notification preferences updated successfully.",
        "user_profile": updated_profile
    }

# --- QUIZ ADMIN/GAMEPLAY ENDPOINTS (UPDATED to reference user profile) ---

@app.post("/admin/set_daily_quiz")
async def set_daily_quiz(quiz_data: DailyQuizData, request: Request):
    # This logic remains in-memory as the quiz state is only for the current day
    daily_quiz_state["quiz_id"] += 1
    daily_quiz_state["quiz_data"] = quiz_data
    daily_quiz_state["start_time"] = datetime.now()
    daily_quiz_state["expiration_time"] = datetime.now() + timedelta(minutes=quiz_data.expiration_minutes)

    global user_quiz_progress
    user_quiz_progress = {} 
    
    return {
        "status": "success",
        "message": f"Daily quiz {daily_quiz_state['quiz_id']} set successfully.",
        "quiz_info": {
            "name": quiz_data.quiz_name,
            "questions_count": len(quiz_data.questions)
        }
    }

@app.post("/quiz/daily_info")
async def get_daily_quiz_info(user_request: TelegramID, db: FirestoreClient = Depends(get_database)):
    """Fetches quiz status, reading user's completion status from Firestore."""
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]
    
    if quiz_data is None:
        return {"status": "no_quiz", "message": "No quiz has been set for today."}

    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(telegram_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_profile = user_doc.to_dict()

    if daily_quiz_state["expiration_time"] < datetime.now():
        return {"status": "expired", "message": "The daily quiz has expired."}

    has_completed = quiz_id in user_profile.get("completed_quizzes", [])
    total_points = len(quiz_data.questions) * quiz_data.points_per_question

    return {
        "status": "available" if not has_completed else "completed",
        "quiz_id": quiz_id,
        "quiz_name": quiz_data.quiz_name,
        "difficulty": quiz_data.difficulty,
        "description": quiz_data.description,
        "questions_count": len(quiz_data.questions),
        "time_limit_seconds": quiz_data.time_limit_seconds,
        "total_points": total_points,
        "expiration_timestamp": daily_quiz_state["expiration_time"].timestamp(),
        "has_completed": has_completed
    }

@app.post("/quiz/start_session")
async def start_quiz_session(user_request: TelegramID, db: FirestoreClient = Depends(get_database)):
    """Starts the quiz session, checking user status against Firestore."""
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]

    if quiz_data is None:
        raise HTTPException(status_code=404, detail="No active quiz available.")

    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(telegram_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found.")
    
    user_profile = user_doc.to_dict()
    
    if quiz_id in user_profile.get("completed_quizzes", []):
        raise HTTPException(status_code=400, detail="You have already completed this quiz.")

    user_quiz_progress[telegram_id] = {
        "quiz_id": quiz_id,
        "current_score": 0,
        "correct_count": 0,
        "answered_questions": {}
    }
    
    first_question = quiz_data.questions[0]
    
    return {
        "status": "session_started",
        "quiz_id": quiz_id,
        "question": first_question.question,
        "q_id": first_question.q_id,
        "options": first_question.options,
        "total_questions": len(quiz_data.questions),
        "next_question_id": first_question.q_id
    }

@app.post("/quiz/answer_question")
async def submit_quiz_answer(submission: AnswerSubmission):
    """Submits the answer (logic remains in-memory progress dict)."""
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]
    
    if quiz_id != submission.quiz_id or quiz_data is None:
        raise HTTPException(status_code=400, detail="Invalid or expired quiz ID.")

    user_progress = user_quiz_progress.get(submission.telegram_id)
    if not user_progress:
        raise HTTPException(status_code=400, detail="Quiz session not started.")

    if submission.question_id in user_progress["answered_questions"]:
        raise HTTPException(status_code=400, detail="Question already answered.")

    current_question = next((q for q in quiz_data.questions if q.q_id == submission.question_id), None)
    if not current_question:
        raise HTTPException(status_code=404, detail="Question not found.")

    is_correct = submission.selected_option_index == current_question.correct_option_index
    
    points = quiz_data.points_per_question if is_correct else 0
    user_progress["current_score"] += points
    if is_correct:
        user_progress["correct_count"] += 1
    
    user_progress["answered_questions"][submission.question_id] = is_correct

    all_q_ids = [q.q_id for q in quiz_data.questions]
    answered_q_ids = user_progress["answered_questions"].keys()
    
    next_q_id = next((q_id for q_id in all_q_ids if q_id not in answered_q_ids), None)
    
    response = {
        "status": "next_question",
        "is_correct": is_correct,
        "points_earned": points,
        "correct_answer": current_question.options[current_question.correct_option_index],
        "next_question_id": next_q_id
    }
    
    if next_q_id is not None:
        next_question = next((q for q in quiz_data.questions if q.q_id == next_q_id), None)
        response["question"] = next_question.question
        response["options"] = next_question.options
    
    return response

@app.post("/quiz/results")
async def finalize_quiz_results(user_request: TelegramID, db: FirestoreClient = Depends(get_database)):
    """
    Finalizes the quiz, updates user's persistent stats, and updates all joined leagues in Firestore.
    """
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    user_progress = user_quiz_progress.get(telegram_id)

    # 1. Fetch User Profile
    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(telegram_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists or not user_progress or user_progress["quiz_id"] != quiz_id:
        raise HTTPException(status_code=400, detail="Invalid quiz session or user data.")
    
    user_profile = user_doc.to_dict()
    league_points_earned = user_progress["current_score"]
    total_questions = len(daily_quiz_state["quiz_data"].questions)
    
    # 2. Update Persistent User Stats
    is_perfect_score = user_progress["correct_count"] == total_questions
    
    user_profile["overall_score"] += user_progress["current_score"]
    user_profile["total_quizzes_answered"] += 1
    user_profile["correct_answers"] += user_progress["correct_count"]
    user_profile["completed_quizzes"].append(quiz_id)

    # Update Streak and Accuracy logic
    user_profile["current_streak"] = user_profile.get("current_streak", 0) + 1 if is_perfect_score else 0
    user_profile["best_streak"] = max(user_profile.get("best_streak", 0), user_profile["current_streak"])

    # Update Accuracy History
    past_accuracy = user_profile.get("past_accuracy", [0, 0, 0])
    past_accuracy[0], past_accuracy[1] = past_accuracy[1], past_accuracy[2]
    today_accuracy = calculate_accuracy(user_progress["correct_count"], total_questions)
    past_accuracy[2] = today_accuracy
    user_profile["past_accuracy"] = past_accuracy
    
    user_profile["accuracy_rate"] = calculate_accuracy(
        user_profile["correct_answers"], 
        user_profile["total_quizzes_answered"]
    )
    
    # 3. Update League Scores (MUST USE FIRESTORE NOW)
    leagues_to_update = user_profile.get("leagues", {}).keys()
    
    for code in leagues_to_update:
        league_ref = db.collection(LEAGUES_COLLECTION_PATH).document(code)
        league_doc = league_ref.get()
        
        if league_doc.exists:
            league = league_doc.to_dict()
            
            # Update user's points reference in their profile
            user_profile["leagues"][code] = user_profile["leagues"].get(code, 0) + league_points_earned
            
            # Update points in the league's persistent member list
            members_list = league.get("members", [])
            
            for member in members_list:
                if member.get("telegram_id") == telegram_id:
                    member["league_points"] = member.get("league_points", 0) + league_points_earned
                    break
            
            # Save the updated league document back to Firestore
            league_ref.update({"members": members_list})

    # 4. Save the updated user profile back to Firestore
    user_ref.set(user_profile)
    
    # 5. Cleanup and Return
    del user_quiz_progress[telegram_id]
    
    return {
        "status": "complete",
        "message": "Quiz completed successfully. Persistent stats updated.",
        "score_earned": user_progress["current_score"],
        "correct_count": user_progress["correct_count"],
        "total_questions": total_questions,
        "user_profile": user_profile
    }

# --- NEW: LEAGUE ENDPOINTS (ALL NOW PERSISTENT) ---

@app.post("/league/create")
async def create_league(league_data: LeagueCreation, db: FirestoreClient = Depends(get_database)):
    """
    Creates a new league and enrolls the creator as the first member, saving it to Firestore.
    """
    creator_id = league_data.telegram_id
    
    # 1. Fetch User Profile (Need username for member list)
    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(creator_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="Creator not found. Please log in again.")
    
    user_profile = user_doc.to_dict()
    creator_username = user_profile.get("username", f"User {creator_id[:4]}...")

    # 2. Generate a unique 6-digit code (checks Firestore)
    code = generate_league_code(db) 

    new_league = {
        "code": code,
        "name": league_data.name,
        "description": league_data.description,
        "difficulty": league_data.difficulty_tag,
        "avatar_url": league_data.league_avatar_url,
        "is_private": league_data.is_private,
        "member_limit": league_data.member_limit,
        "start_date": league_data.start_date,
        "owner_id": creator_id,
        "members": [
            {"telegram_id": creator_id, "league_points": 0, "username": creator_username}
        ],
        "member_ids": [creator_id] # Helper list for quick lookups
    }
    
    # 3. Save the league to persistent Firestore
    league_ref = db.collection(LEAGUES_COLLECTION_PATH).document(code)
    league_ref.set(new_league)
    
    # 4. Update the creator's profile in Firestore (Add league reference)
    user_profile["leagues"] = user_profile.get("leagues", {})
    user_profile["leagues"][code] = 0 # Initialize with 0 points
    user_ref.update({"leagues": user_profile["leagues"]})
    
    return {
        "status": "success",
        "message": f"League '{league_data.name}' created successfully.",
        "join_code": code,
        "league_details": new_league
    }

@app.post("/league/join")
async def join_league(join_data: LeagueJoin, db: FirestoreClient = Depends(get_database)):
    """Allows a user to join a league, updating both user and league documents in Firestore."""
    user_id = join_data.telegram_id
    code = join_data.code.upper() 

    # 1. Fetch User Profile
    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found. Please log in again.")
    user_profile = user_doc.to_dict()
    joiner_username = user_profile.get("username", f"User {user_id[:4]}")

    # 2. Fetch League Document
    league_ref = db.collection(LEAGUES_COLLECTION_PATH).document(code)
    league_doc = league_ref.get()
    
    if not league_doc.exists:
        raise HTTPException(status_code=404, detail="League code is invalid or expired.")

    league = league_doc.to_dict()
    members = league.get("members", [])
    member_ids = league.get("member_ids", [])
    
    # Check if league is full
    if len(members) >= league.get("member_limit", 50):
        raise HTTPException(status_code=403, detail="League is full.")
        
    # Check if user is already a member
    if user_id in member_ids:
         return {"status": "success", "message": "You are already a member of this league."}
        
    # 3. Perform Updates
    
    # Add user to league (Denormalize username)
    members.append({"telegram_id": user_id, "league_points": 0, "username": joiner_username})
    member_ids.append(user_id)
    
    # Add league to user's persistent profile
    user_profile["leagues"] = user_profile.get("leagues", {})
    user_profile["leagues"][code] = 0
    
    # 4. Save Changes to Firestore
    league_ref.update({"members": members, "member_ids": member_ids})
    user_ref.update({"leagues": user_profile["leagues"]})
    
    return {
        "status": "success",
        "message": f"Successfully joined league '{league['name']}'.",
        "league_details": league
    }

@app.post("/league/my_leagues")
async def get_my_leagues(user_request: TelegramID, db: FirestoreClient = Depends(get_database)):
    """Returns the list of leagues the user belongs to, fetching data from Firestore."""
    user_id = user_request.telegram_id
    
    # 1. Fetch User Profile to get league list
    user_ref = db.collection(USERS_PROFILE_COLLECTION_PATH).document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_profile = user_doc.to_dict()
    my_league_codes = user_profile.get("leagues", {}).keys()
    my_leagues_list = []
    
    # 2. Batch fetch league details
    for code in my_league_codes:
        league_ref = db.collection(LEAGUES_COLLECTION_PATH).document(code)
        league_doc = league_ref.get()
        
        if league_doc.exists:
            league = league_doc.to_dict()
            members = league.get("members", [])
            
            # Calculate Rank
            sorted_members = sorted(members, key=lambda m: m.get("league_points", 0), reverse=True)
            user_rank = next((i + 1 for i, m in enumerate(sorted_members) if m.get("telegram_id") == user_id), "N/A")
            
            my_leagues_list.append({
                "league_avatar_url": league.get("avatar_url"),
                "league_name": league["name"],
                "league_description": league["description"],
                "user_rank": user_rank,
                "user_points": user_profile["leagues"].get(code, 0),
                "member_count": len(members),
                "is_owner": league.get("owner_id") == user_id,
                "code": code
            })
            
    return {
        "status": "success",
        "my_leagues": my_leagues_list
    }
    
@app.get("/league/discover")
async def discover_leagues(db: FirestoreClient = Depends(get_database)):
    """Returns 3 random public leagues, querying from Firestore."""
    
    # Query for public leagues (where is_private is False)
    # Note: Firestore query is limited. We'll fetch a small set and randomly select from that.
    query = db.collection(LEAGUES_COLLECTION_PATH).where("is_private", "==", False).limit(10)
    public_leagues_docs = query.stream()
    
    public_leagues = [doc.to_dict() for doc in public_leagues_docs]
    
    if not public_leagues:
        return {"status": "success", "public_leagues": [], "message": "No public leagues available."}

    # Select up to 3 random leagues
    random_leagues = random.sample(public_leagues, min(3, len(public_leagues)))
    
    # Format the data for the frontend display
    discovery_list = []
    for league in random_leagues:
        discovery_list.append({
            "league_avatar_url": league.get("avatar_url"),
            "league_name": league["name"],
            "league_description": league["description"],
            "league_difficulty": league["difficulty"],
            "member_count": len(league.get("members", [])),
            "join_code": league["code"] 
        })
        
    return {
        "status": "success",
        "public_leagues": discovery_list
    }

@app.post("/league/search")
async def search_leagues(search_data: LeagueSearch, db: FirestoreClient = Depends(get_database)):
    """Allows users to search for public leagues by name, using Firestore."""
    query_text = search_data.query.strip()
    
    # Note: Full-text search is complex in Firestore. 
    # We will use an equality filter for a simple, case-insensitive match on exact names 
    # or rely on the frontend to search through a list of public leagues if the dataset is small.
    # For now, we fetch all public leagues and filter locally (scalable only for small apps).
    
    all_public_leagues = []
    public_query = db.collection(LEAGUES_COLLECTION_PATH).where("is_private", "==", False).stream()
    all_public_leagues = [doc.to_dict() for doc in public_query]
    
    matching_leagues = [
        l for l in all_public_leagues
        if query_text.lower() in l["name"].lower()
    ]
    
    # Format the data for the frontend display (similar to discover)
    search_results = []
    for league in matching_leagues:
        search_results.append({
            "league_avatar_url": league.get("avatar_url"),
            "league_name": league["name"],
            "league_description": league["description"],
            "league_difficulty": league["difficulty"],
            "member_count": len(league.get("members", [])),
            "join_code": league["code"]
        })

    return {
        "status": "success",
        "search_results": search_results
    }
    
@app.post("/api/league/leaderboard")
async def get_league_leaderboard(request_data: LeagueDetailsRequest, db: FirestoreClient = Depends(get_database)):
    """
    Fetches the full member list for a specific league from persistent Firestore storage.
    """
    user_id = request_data.telegram_id
    code = request_data.league_id.upper()
    
    league_ref = db.collection(LEAGUES_COLLECTION_PATH).document(code)
    
    try:
        # Check Firestore for the league document
        league_doc = league_ref.get()
        
        if not league_doc.exists:
            # This check now points to persistent storage, permanently fixing the 404 issue!
            raise HTTPException(status_code=404, detail="League not found in persistent database.")

        league = league_doc.to_dict()

    except Exception as e:
        print(f"Database error fetching league {code}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during data retrieval.")

    # Authorization Check: Ensure the user is a member of the league (or the league is public)
    member_ids = league.get("member_ids", [])
    is_member = user_id in member_ids
    
    if league.get("is_private") and not is_member:
        raise HTTPException(status_code=403, detail="Not authorized to view this private league's leaderboard.")

    # 1. Sort members (using denormalized points)
    members: List[Dict[str, Any]] = league.get("members", [])
    leaderboard = sorted(
        members, 
        key=lambda m: m.get("league_points", 0), 
        reverse=True
    )

    # 2. Format output for the frontend
    leaderboard_with_names = []
    for rank, member in enumerate(leaderboard, 1):
        member_id = member.get("telegram_id")
        
        leaderboard_with_names.append({
            "rank": rank,
            "telegram_id": member_id,
            "username": member.get("username", f"User {member_id[:4]}..."), # Use denormalized username
            "points": member.get("league_points", 0),
            "is_current_user": member_id == user_id
        })

    return {
        "status": "success",
        "league_name": league.get("name", "Unknown League"),
        "league_code": code,
        "leaderboard": leaderboard_with_names
    }

# --- STATIC FILES MOUNT ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")
