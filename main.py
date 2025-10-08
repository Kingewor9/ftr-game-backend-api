import hmac
import hashlib
import json
import asyncio # <-- NEW: Required for asynchronous operations (like updating leagues)
from urllib.parse import unquote_plus
from fastapi import FastAPI, Request, HTTPException, Body
from pydantic import BaseModel, Field
from urllib.parse import unquote_plus, parse_qsl 
from datetime import datetime, timedelta
import os
import random
import logging
import string
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.staticfiles import StaticFiles
from typing import Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient # <-- NEW: MongoDB Driver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# CRITICAL: LOADED FROM ENVIRONMENT VARIABLE
BOT_TOKEN: Optional [str] = os.getenv ("BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("[FATAL ERROR] Cannot proceed: BOT_TOKEN environment variable is NOT set. Hash validation will fail.")
else:
    logger.info(f"[DIAGNOSTIC] BOT_TOKEN successfully loaded. Length: {len(BOT_TOKEN)}.")

# Define the admin's Telegram ID for exclusive access to admin endpoints
ADMIN_TELEGRAM_ID = "1474715816"

# --- MongoDB Configuration ---
# NOTE: Set the MONGO_URI environment variable or update the default here
MONGO_DETAILS = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "telegram_quiz_game"
USER_COLLECTION = "user_profiles"
LEAGUE_COLLECTION = "leagues"

# Global MongoDB Variables (Initialized in startup event)
client: Optional[AsyncIOMotorClient] = None
user_profiles_collection: Any = None
league_collection: Any = None

# --- Pydantic Data Models ---

# Pydantic Model representing the data structure in MongoDB
class UserProfileDB(BaseModel):
    # Field(..., alias="_id") maps MongoDB's internal ID to a Python field (only used internally)
    id: Optional[str] = Field(None, alias="_id") 
    telegram_id: str
    username: str
    email: str | None = None
    avatar_url: str | None = None
    member_since: str
    overall_score: int
    total_quizzes_answered: int
    correct_answers: int
    accuracy_rate: int
    current_streak: int
    best_streak: int
    leagues: dict[str, int] # {code: points} map
    past_accuracy: list[int]
    preferences: dict
    completed_quizzes: list[int]
    
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
    questions: list[QuizQuestion]

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
    
# --- LEAGUE MODELS (UNCHANGED) ---

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
    
class LeagueDetailsRequest(BaseModel):
    telegram_id: str = Field(..., description="The user requesting the leaderboard.")
    league_id: str = Field(..., description="The 6-digit code/ID of the league to view.")


# --- Temporary "Database" (REMAINING IN-MEMORY STATE) ---
# This holds ephemeral game state and does not require MongoDB persistence
daily_quiz_state = {
    "quiz_id": 0,
    "quiz_data": None,
    "start_time": None,
    "expiration_time": None
}
user_quiz_progress = {} 

# --- Helper Functions ---

# !!! CORRECTED AUTHENTICATION LOGIC !!! (UNCHANGED)
def validate_telegram_data(init_data: str) -> dict:
    """
    Validates the hash of the received Telegram Mini App init data.
    """
    
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
        raise HTTPException(status_code=400, detail="Missing essential Telegram data (hash or user).")
        
    data_check_string_parts.sort()
    data_check_string = "\n".join(data_check_string_parts)
    
    # Calculate the expected hash
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

    # Compare hashes
    if calculated_hash != hash_value:
        print(f"[ERROR] Hash Mismatch Detected! Calculated: {calculated_hash}, Received: {hash_value}")
        raise HTTPException(status_code=403, detail="Invalid Telegram data hash.")
    
    print(f"[SUCCESS] Hash validated successfully: {calculated_hash}")

    # Extract user info
    try:
        user_info = json.loads(unquote_plus(user_data_str))
        return user_info
    except json.JSONDecodeError as e:
        print(f"JSON decode error in user data: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in user data.")


def calculate_accuracy(correct, total):
    return round((correct / total) * 100) if total > 0 else 0

async def generate_league_code(length=6):
    """Generates a unique 6-digit alphanumeric code by checking MongoDB."""
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choice(chars) for _ in range(length))
        # --- MongoDB Read (Check for uniqueness) ---
        if not await league_collection.find_one({"code": code}):
            return code

# --- API Setup ---
app = FastAPI()

# =======================================================================
# >>> MONGODB CONNECTION HOOKS (NEW) <<<
# =======================================================================

@app.on_event("startup")
async def startup_db_client():
    """Initializes MongoDB connection and collections on FastAPI startup."""
    global client, user_profiles_collection, league_collection
    try:
        client = AsyncIOMotorClient(MONGO_DETAILS)
        database = client[DATABASE_NAME]
        
        user_profiles_collection = database[USER_COLLECTION]
        league_collection = database[LEAGUE_COLLECTION]

        # Ensure unique index on telegram_id for fast user lookups
        await user_profiles_collection.create_index("telegram_id", unique=True)
        # Ensure unique index on code for leagues
        await league_collection.create_index("code", unique=True)

        logger.info(f"MongoDB connected successfully to DB: {DATABASE_NAME}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Closes MongoDB connection on FastAPI shutdown."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

# =======================================================================
# >>> CRITICAL CORS FIX <<<
# =======================================================================
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# =======================================================================
# >>> HEALTH CHECK ENDPOINT <<<
# =======================================================================
@app.get("/api/status")
async def health_check():
    """A simple endpoint to verify the API server is alive."""
    return {"status": "ok", "message": "API is running and accessible."}

# =======================================================================

# --- CORE API Endpoints (Login and Profile - MONGODB IMPLEMENTATION) ---

@app.post("/auth/login")
async def telegram_login(request: Request):
    """Handles TWA authentication and user profile retrieval/creation via MongoDB."""
    try:
        body = await request.body()
        init_data = body.decode('utf-8')
        
        user_info = validate_telegram_data(init_data)
        telegram_id = str(user_info.get("id"))
        
        # --- MongoDB Read Operation: Check for existing user ---
        user_profile_doc = await user_profiles_collection.find_one({"telegram_id": telegram_id})

        if user_profile_doc:
            # User exists
            user_profile = UserProfileDB(**user_profile_doc).dict(by_alias=True)
            user_profile.pop('_id', None) # Remove MongoDB internal ID for clean response
            logger.info(f"User authenticated: {telegram_id}")
        else:
            # New User, create profile
            username = user_info.get("username") or user_info.get("first_name")
            
            new_user_data = {
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
                "leagues": {},
                "past_accuracy": [0, 0, 0], 
                "preferences": NotificationPreferences().dict(),
                "completed_quizzes": [] 
            }
            
            # --- MongoDB Write Operation (Insert) ---
            await user_profiles_collection.insert_one(new_user_data)
            logger.info(f"New user created: {telegram_id}")
            user_profile = new_user_data
            
        return {
            "status": "success",
            "message": "User authenticated and profile retrieved.",
            "user_profile": user_profile,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An error occurred during login: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.put("/profile/edit")
async def edit_profile(profile_data: UserProfileEdit):
    user_id = profile_data.telegram_id
    
    # --- MongoDB Update Operation ---
    update_data = profile_data.dict(exclude_unset=True)
    update_data.pop('telegram_id', None)

    result = await user_profiles_collection.update_one(
        {"telegram_id": user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found.")
        
    # Fetch updated profile to send back
    updated_doc = await user_profiles_collection.find_one({"telegram_id": user_id})
    updated_profile = UserProfileDB(**updated_doc).dict(by_alias=True)
    updated_profile.pop('_id', None)
    
    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "user_profile": updated_profile
    }

@app.put("/profile/preferences")
async def update_preferences(prefs_data: UserPreferencesUpdate):
    user_id = prefs_data.telegram_id
    
    # --- MongoDB Update Operation ---
    result = await user_profiles_collection.update_one(
        {"telegram_id": user_id},
        {"$set": {"preferences": prefs_data.preferences.dict()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found.")
        
    # Fetch updated profile
    updated_doc = await user_profiles_collection.find_one({"telegram_id": user_id})
    updated_profile = UserProfileDB(**updated_doc).dict(by_alias=True)
    updated_profile.pop('_id', None)
    
    return {
        "status": "success",
        "message": "Notification preferences updated successfully.",
        "user_profile": updated_profile
    }

# --- QUIZ ADMIN/GAMEPLAY ENDPOINTS ---

@app.post("/admin/set_daily_quiz")
async def set_daily_quiz(quiz_data: DailyQuizData, request: Request):
    # NOTE: This endpoint remains in-memory for the quiz state
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
async def get_daily_quiz_info(user_request: TelegramID):
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]
    
    if quiz_data is None:
        return {"status": "no_quiz", "message": "No quiz has been set for today."}

    # --- MongoDB Read (User profile) ---
    user_profile_doc = await user_profiles_collection.find_one({"telegram_id": telegram_id})
    if not user_profile_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    user_profile = UserProfileDB(**user_profile_doc).dict(by_alias=True)

    if daily_quiz_state["expiration_time"] < datetime.now():
        return {"status": "expired", "message": "The daily quiz has expired."}

    has_completed = quiz_id in user_profile["completed_quizzes"]
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
async def start_quiz_session(user_request: TelegramID):
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]

    if quiz_data is None:
        raise HTTPException(status_code=404, detail="No active quiz available.")

    # Check for user existence and quiz completion status via MongoDB
    user_doc = await user_profiles_collection.find_one({"telegram_id": telegram_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    if quiz_id in user_doc.get("completed_quizzes", []):
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
    # This remains in-memory for the current quiz session
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
async def finalize_quiz_results(user_request: TelegramID):
    """
    Finalizes the quiz, updates user's permanent stats, and updates league scores in MongoDB.
    """
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    user_progress = user_quiz_progress.get(telegram_id)

    # --- MongoDB Read (User Profile) ---
    user_profile_doc = await user_profiles_collection.find_one({"telegram_id": telegram_id})
    if not user_profile_doc or not user_progress or user_progress["quiz_id"] != quiz_id:
        raise HTTPException(status_code=400, detail="Invalid quiz session or user data.")

    user_profile = UserProfileDB(**user_profile_doc).dict(by_alias=True)
    league_points_earned = user_progress["current_score"]

    # 1. Update Permanent User Stats (in-memory calculation)
    is_perfect_score = user_progress["correct_count"] == len(daily_quiz_state["quiz_data"].questions)
    
    user_profile["overall_score"] += user_progress["current_score"]
    user_profile["total_quizzes_answered"] += 1
    user_profile["correct_answers"] += user_progress["correct_count"]
    user_profile["completed_quizzes"].append(quiz_id)

    # 2. Update Streak
    user_profile["current_streak"] = user_profile["current_streak"] + 1 if is_perfect_score else 0 
    if user_profile["current_streak"] > user_profile["best_streak"]:
        user_profile["best_streak"] = user_profile["current_streak"]
        
    # 3. Update Accuracy and History
    user_profile["past_accuracy"].pop(0)
    today_accuracy = calculate_accuracy(
        user_progress["correct_count"], 
        len(daily_quiz_state["quiz_data"].questions)
    )
    user_profile["past_accuracy"].append(today_accuracy)
    
    user_profile["accuracy_rate"] = calculate_accuracy(
        user_profile["correct_answers"], 
        user_profile["total_quizzes_answered"] * len(daily_quiz_state["quiz_data"].questions)
    )
    
    # 4. Update League Scores in MongoDB
    league_update_tasks = []
    league_map_updates = {}
    
    for code in user_profile["leagues"].keys():
        user_profile["leagues"][code] += league_points_earned
        league_map_updates[f"leagues.{code}"] = user_profile["leagues"][code]
        
        # MongoDB Update: Update the points in the league collection's member array
        league_update_tasks.append(
            league_collection.update_one(
                {"code": code, "members.telegram_id": telegram_id},
                {"$inc": {"members.$.league_points": league_points_earned}}
            )
        )
    
    # Run all league updates concurrently
    if league_update_tasks:
        await asyncio.gather(*league_update_tasks)
    
    # 5. MongoDB Write: Save all computed user stats
    update_fields = {
        "overall_score": user_profile["overall_score"],
        "total_quizzes_answered": user_profile["total_quizzes_answered"],
        "correct_answers": user_profile["correct_answers"],
        "accuracy_rate": user_profile["accuracy_rate"],
        "current_streak": user_profile["current_streak"],
        "best_streak": user_profile["best_streak"],
        "past_accuracy": user_profile["past_accuracy"],
        "completed_quizzes": user_profile["completed_quizzes"],
        **league_map_updates
    }
    
    await user_profiles_collection.update_one(
        {"telegram_id": telegram_id},
        {"$set": update_fields}
    )

    # 6. Cleanup and Return
    del user_quiz_progress[telegram_id]
    user_profile.pop('_id', None)
    
    return {
        "status": "complete",
        "message": "Quiz completed successfully. Stats updated.",
        "score_earned": user_progress["current_score"],
        "correct_count": user_progress["correct_count"],
        "total_questions": len(daily_quiz_state["quiz_data"].questions),
        "user_profile": user_profile
    }

# --- LEAGUE ENDPOINTS (MONGODB IMPLEMENTATION) ---

@app.post("/league/create")
async def create_league(league_data: LeagueCreation):
    creator_id = league_data.telegram_id
    
    # --- MongoDB Read (User check) ---
    if not await user_profiles_collection.find_one({"telegram_id": creator_id}):
        raise HTTPException(status_code=404, detail="Creator not found.")

    # Generate a unique 6-digit code (checks DB)
    code = await generate_league_code()

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
            {"telegram_id": creator_id, "league_points": 0}
        ]
    }
    
    # --- MongoDB Write (Insert League) ---
    await league_collection.insert_one(new_league)
    
    # --- MongoDB Write (Update Creator's profile) ---
    await user_profiles_collection.update_one(
        {"telegram_id": creator_id},
        {"$set": {f"leagues.{code}": 0}}
    )
    
    new_league.pop('_id', None) 
    
    return {
        "status": "success",
        "message": f"League '{league_data.name}' created successfully.",
        "join_code": code,
        "league_details": new_league
    }

@app.post("/league/join")
async def join_league(join_data: LeagueJoin):
    user_id = join_data.telegram_id
    code = join_data.code.upper()
    
    # --- MongoDB Read (User check) ---
    user_profile_doc = await user_profiles_collection.find_one({"telegram_id": user_id})
    if not user_profile_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    
    user_profile = UserProfileDB(**user_profile_doc).dict(by_alias=True)

    # --- MongoDB Read (League check) ---
    league_doc = await league_collection.find_one({"code": code})
    if not league_doc:
        raise HTTPException(status_code=404, detail="League code is invalid or expired.")

    league = league_doc
    
    if code in user_profile["leagues"]:
        return {"status": "success", "message": "You are already a member of this league."}
        
    if len(league["members"]) >= league["member_limit"]:
        raise HTTPException(status_code=403, detail="League is full. The join code has expired.")
        
    # --- MongoDB Write (Update League: Add member) ---
    await league_collection.update_one(
        {"code": code},
        {"$push": {"members": {"telegram_id": user_id, "league_points": 0}}}
    )
    
    # --- MongoDB Write (Update User: Add league) ---
    await user_profiles_collection.update_one(
        {"telegram_id": user_id},
        {"$set": {f"leagues.{code}": 0}}
    )
    
    updated_league_doc = await league_collection.find_one({"code": code})
    updated_league_doc.pop('_id', None)
    
    return {
        "status": "success",
        "message": f"Successfully joined league '{league['name']}'.",
        "league_details": updated_league_doc
    }

@app.post("/league/my_leagues")
async def get_my_leagues(user_request: TelegramID):
    user_id = user_request.telegram_id
    
    # --- MongoDB Read (User profile) ---
    user_profile_doc = await user_profiles_collection.find_one({"telegram_id": user_id})
    if not user_profile_doc:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_profile = UserProfileDB(**user_profile_doc).dict(by_alias=True)
    
    my_leagues_list = []
    league_codes = list(user_profile["leagues"].keys())
    
    # --- MongoDB Read (Fetch all leagues the user belongs to) ---
    if league_codes:
        cursor = league_collection.find({"code": {"$in": league_codes}})
        leagues = await cursor.to_list(length=None)
    else:
        leagues = []
        
    for league in leagues:
        code = league["code"]
        
        # 1. Calculate Rank (In-memory after fetch)
        sorted_members = sorted(league["members"], key=lambda m: m["league_points"], reverse=True)
        user_rank = next((i + 1 for i, m in enumerate(sorted_members) if m["telegram_id"] == user_id), "N/A")
        
        # 2. Extract needed details
        my_leagues_list.append({
            "league_avatar_url": league.get("avatar_url"),
            "league_name": league["name"],
            "league_description": league["description"],
            "user_rank": user_rank,
            "user_points": user_profile["leagues"][code],
            "member_count": len(league["members"]),
            "is_owner": league["owner_id"] == user_id,
            "code": code
        })
            
    return {
        "status": "success",
        "my_leagues": my_leagues_list
    }

@app.get("/league/discover")
async def discover_leagues():
    # --- MongoDB Read (Find 3 random public leagues using aggregation) ---
    pipeline = [
        {"$match": {"is_private": False}},
        {"$sample": {"size": 3}}
    ]
    cursor = league_collection.aggregate(pipeline)
    random_leagues = await cursor.to_list(length=None)
    
    if not random_leagues:
        return {"status": "success", "public_leagues": [], "message": "No public leagues available."}

    discovery_list = []
    for league in random_leagues:
        discovery_list.append({
            "league_avatar_url": league.get("avatar_url"),
            "league_name": league["name"],
            "league_description": league["description"],
            "league_difficulty": league["difficulty"],
            "member_count": len(league["members"]),
            "join_code": league["code"]
        })
        
    return {
        "status": "success",
        "public_leagues": discovery_list
    }

@app.post("/league/search")
async def search_leagues(search_data: LeagueSearch):
    query = search_data.query.strip()
    
    # --- MongoDB Read (Case-insensitive search on public leagues) ---
    cursor = league_collection.find({
        "is_private": False,
        "name": {"$regex": query, "$options": "i"} # "i" for case-insensitive
    })
    matching_leagues = await cursor.to_list(length=None)
    
    search_results = []
    for league in matching_leagues:
        search_results.append({
            "league_avatar_url": league.get("avatar_url"),
            "league_name": league["name"],
            "league_description": league["description"],
            "league_difficulty": league["difficulty"],
            "member_count": len(league["members"]),
            "join_code": league["code"]
        })

    return {
        "status": "success",
        "search_results": search_results
    }
    
@app.post("/api/league/leaderboard")
async def get_league_leaderboard(request_data: LeagueDetailsRequest):
    user_id = request_data.telegram_id
    code = request_data.league_id.upper() 

    # --- MongoDB Read (League) ---
    league_doc = await league_collection.find_one({"code": code})
    if not league_doc:
        raise HTTPException(status_code=404, detail="League not found.")

    league = league_doc

    is_member = any(member["telegram_id"] == user_id for member in league["members"])
    if league["is_private"] and not is_member:
        raise HTTPException(status_code=403, detail="Not authorized to view this private league's leaderboard.")

    # 1. Sort members (In-memory)
    leaderboard = sorted(
        league["members"], 
        key=lambda m: m["league_points"], 
        reverse=True
    )
    
    # 2. Collect IDs for batch username lookup
    member_ids = [member["telegram_id"] for member in leaderboard]
    
    # --- MongoDB Read (Batch Usernames) ---
    cursor = user_profiles_collection.find(
        {"telegram_id": {"$in": member_ids}},
        {"telegram_id": 1, "username": 1, "_id": 0}
    )
    user_map = {doc["telegram_id"]: doc["username"] for doc in await cursor.to_list(length=None)}

    # 3. Enhance the output for the frontend
    leaderboard_with_names = []
    for rank, member in enumerate(leaderboard, 1):
        member_id = member["telegram_id"]
        username = user_map.get(member_id, f"User {member_id[:4]}...")
        
        leaderboard_with_names.append({
            "rank": rank,
            "telegram_id": member_id,
            "username": username,
            "points": member["league_points"],
            "is_current_user": member_id == user_id
        })

    return {
        "status": "success",
        "league_name": league["name"],
        "league_code": code,
        "leaderboard": leaderboard_with_names
    }

# 2. STATIC FILES MOUNT
app.mount("/", StaticFiles(directory="static", html=True), name="static")
