import hmac
import hashlib
import json
from urllib.parse import unquote_plus
from fastapi import FastAPI, Request, HTTPException, Body
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import os
import random
import string
# NEW IMPORT REQUIRED FOR CORS FIX
from fastapi.middleware.cors import CORSMiddleware 
# >>> REQUIRED IMPORT FOR SERVING HTML FRONTEND <<<
from fastapi.staticfiles import StaticFiles 



# --- Configuration ---
# Replace with your actual bot token
BOT_TOKEN = "8384275400:AAGcl3BVdDx5Qo0S-uD8Jy1jynWB_gwGNMQ" 
# Define the admin's Telegram ID for exclusive access to admin endpoints
ADMIN_TELEGRAM_ID = "1474715816" 
# ---

if BOT_TOKEN == "8384275400:AAGcl3BVdDx5Qo0S-uD8Jy1jynWB_gwGNMQ":
    raise ValueError("Please replace 'YOUR_TELEGRAM_BOT_TOKEN' with your actual bot token.")
if ADMIN_TELEGRAM_ID == "YOUR_ADMIN_TELEGRAM_ID":
    print("WARNING: ADMIN_TELEGRAM_ID is set to placeholder. Admin features will be restricted.")


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

# --- Temporary "Database" ---
user_db = {} 

daily_quiz_state = {
    "quiz_id": 0,
    "quiz_data": None,
    "start_time": None,
    "expiration_time": None
}
user_quiz_progress = {} 

# NEW LEAGUE DATABASE: Key is the 6-digit code
league_db = {}
# Example League Structure:
# "ABC123": {
#     "code": "ABC123",
#     "name": "My Pro League",
#     "owner_id": "12345",
#     "is_private": True,
#     "member_limit": 50,
#     "members": [
#         {"telegram_id": "12345", "league_points": 100},
#         {"telegram_id": "67890", "league_points": 50},
#     ]
# }


# --- Helper Functions ---

def validate_telegram_data(init_data: str) -> dict:
    # (UNCHANGED validation logic)
    data_check_string_parts = []
    hash_value = ""
    
    init_data_list = sorted([item for item in init_data.split('&') if not item.startswith('hash')])
    
    for item in init_data_list:
        if item.startswith('hash='):
            hash_value = item[5:]
        else:
            data_check_string_parts.append(item)
    
    data_check_string = "\n".join(data_check_string_parts)
    
    secret_key = hashlib.sha256(BOT_TOKEN.encode()).digest()
    
    calculated_hash = hmac.new(
        secret_key,
        msg=data_check_string.encode(),
        digestmod=hashlib.sha256
    ).hexdigest()

    if calculated_hash != hash_value:
        raise HTTPException(status_code=403, detail="Invalid Telegram data hash.")

    user_data_str = ""
    for part in init_data.split('&'):
        if part.startswith('user='):
            user_data_str = unquote_plus(part[5:])
            break
            
    if not user_data_str:
        raise HTTPException(status_code=400, detail="User data not found in initData.")
    
    try:
        user_info = json.loads(user_data_str)
        return user_info
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in user data.")

def calculate_accuracy(correct, total):
    return round((correct / total) * 100) if total > 0 else 0

def generate_league_code(length=6):
    """Generates a unique 6-digit alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choice(chars) for _ in range(length))
        if code not in league_db:
            return code

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

# --- CORE API Endpoints (Login and Profile - UNCHANGED) ---

@app.post("/auth/login")
async def telegram_login(request: Request):
    # ... (Login logic)
    try:
        body = await request.body()
        init_data = body.decode('utf-8')
        
        user_info = validate_telegram_data(init_data)
        telegram_id = str(user_info.get("id"))
        
        if telegram_id not in user_db:
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
                "leagues": {}, # MODIFIED: Leagues is now a dict {code: league_points}
                "past_accuracy": [0, 0, 0], 
                "preferences": NotificationPreferences().dict(),
                "completed_quizzes": [] 
            }
            user_db[telegram_id] = new_user
            print(f"New user created: {username} (ID: {telegram_id})")
        
        return {
            "status": "success",
            "message": "User authenticated and profile retrieved.",
            "user_profile": user_db[telegram_id],
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.put("/profile/edit")
async def edit_profile(profile_data: UserProfileEdit):
    # ... (Existing edit_profile logic)
    user_id = profile_data.telegram_id
    
    if user_id not in user_db:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_db[user_id]["username"] = profile_data.username
    user_db[user_id]["email"] = profile_data.email
    user_db[user_id]["avatar_url"] = profile_data.avatar_url
    
    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "user_profile": user_db[user_id]
    }

@app.put("/profile/preferences")
async def update_preferences(prefs_data: UserPreferencesUpdate):
    # ... (Existing update_preferences logic)
    user_id = prefs_data.telegram_id
    
    if user_id not in user_db:
        raise HTTPException(status_code=404, detail="User not found.")
        
    user_db[user_id]["preferences"] = prefs_data.preferences.dict()
    
    return {
        "status": "success",
        "message": "Notification preferences updated successfully.",
        "user_profile": user_db[user_id]
    }

# --- QUIZ ADMIN/GAMEPLAY ENDPOINTS (UNCHANGED except for leagues logic in results) ---

@app.post("/admin/set_daily_quiz")
async def set_daily_quiz(quiz_data: DailyQuizData, request: Request):
    # ... (Admin logic - UNCHANGED)
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

class TelegramID(BaseModel):
    telegram_id: str = Field(..., description="The unique ID of the Telegram user.")

@app.post("/quiz/daily_info")
async def get_daily_quiz_info(user_request: TelegramID):
    # ... (Quiz info logic - UNCHANGED)
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]
    
    if quiz_data is None:
        return {"status": "no_quiz", "message": "No quiz has been set for today."}

    user_profile = user_db.get(telegram_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User not found.")

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
    # ... (Quiz start session logic - UNCHANGED)
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    quiz_data = daily_quiz_state["quiz_data"]

    if quiz_data is None:
        raise HTTPException(status_code=404, detail="No active quiz available.")

    user_profile = user_db.get(telegram_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User not found.")

    if quiz_id in user_profile["completed_quizzes"]:
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
    # ... (Quiz answer logic - UNCHANGED)
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
    Finalizes the quiz, updates user's permanent stats, and prepares for 'Back to Quiz' button.
    """
    telegram_id = user_request.telegram_id
    quiz_id = daily_quiz_state["quiz_id"]
    user_profile = user_db.get(telegram_id)
    user_progress = user_quiz_progress.get(telegram_id)

    if not user_profile or not user_progress or user_progress["quiz_id"] != quiz_id:
        raise HTTPException(status_code=400, detail="Invalid quiz session or user data.")
    
    # Calculate league points (just use the current score for now)
    league_points_earned = user_progress["current_score"]

    # 1. Update Permanent User Stats
    is_perfect_score = user_progress["correct_count"] == len(daily_quiz_state["quiz_data"].questions)
    
    user_profile["overall_score"] += user_progress["current_score"]
    user_profile["total_quizzes_answered"] += 1
    user_profile["correct_answers"] += user_progress["correct_count"]
    user_profile["completed_quizzes"].append(quiz_id)

    # 2. Update Streak
    if is_perfect_score:
        user_profile["current_streak"] += 1
    else:
        user_profile["current_streak"] = 0 
    
    if user_profile["current_streak"] > user_profile["best_streak"]:
        user_profile["best_streak"] = user_profile["current_streak"]
        
    # 3. Update Accuracy and History
    yesterday_accuracy = user_profile["past_accuracy"][2] 
    
    user_profile["past_accuracy"][0] = user_profile["past_accuracy"][1]
    user_profile["past_accuracy"][1] = yesterday_accuracy
    
    today_accuracy = calculate_accuracy(
        user_progress["correct_count"], 
        len(daily_quiz_state["quiz_data"].questions)
    )
    user_profile["past_accuracy"][2] = today_accuracy
    
    user_profile["accuracy_rate"] = calculate_accuracy(
        user_profile["correct_answers"], 
        user_profile["total_quizzes_answered"]
    )
    
    # 4. NEW: Update League Scores
    for code in user_profile["leagues"].keys():
        if code in league_db:
            # Update points in user's profile dict
            user_profile["leagues"][code] += league_points_earned
            
            # Update points in the league_db member list
            league = league_db[code]
            for member in league["members"]:
                if member["telegram_id"] == telegram_id:
                    member["league_points"] += league_points_earned
                    break

    # 5. Cleanup and Return
    del user_quiz_progress[telegram_id]
    
    return {
        "status": "complete",
        "message": "Quiz completed successfully. Stats updated.",
        "score_earned": user_progress["current_score"],
        "correct_count": user_progress["correct_count"],
        "total_questions": len(daily_quiz_state["quiz_data"].questions),
        "user_profile": user_profile
    }

# --- NEW: LEAGUE ENDPOINTS ---

@app.post("/league/create")
async def create_league(league_data: LeagueCreation):
    """
    Creates a new league and enrolls the creator as the first member.
    """
    creator_id = league_data.telegram_id
    if creator_id not in user_db:
        raise HTTPException(status_code=404, detail="Creator not found.")

    # Generate a unique 6-digit code
    code = generate_league_code()

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
    
    league_db[code] = new_league
    
    # Add league to creator's profile
    user_db[creator_id]["leagues"][code] = 0 # Initialize with 0 points
    
    return {
        "status": "success",
        "message": f"League '{league_data.name}' created successfully.",
        "join_code": code,
        "league_details": new_league
    }

@app.post("/league/join")
async def join_league(join_data: LeagueJoin):
    """
    Allows a user to join a private league via code or a public league via discovery.
    """
    user_id = join_data.telegram_id
    code = join_data.code.upper() # Ensure case-insensitivity for the code
    
    if user_id not in user_db:
        raise HTTPException(status_code=404, detail="User not found.")
        
    if code not in league_db:
        raise HTTPException(status_code=404, detail="League code is invalid or expired.")

    league = league_db[code]
    
    # Check if league is full
    if len(league["members"]) >= league["member_limit"]:
        raise HTTPException(status_code=403, detail="League is full. The join code has expired.")
        
    # Check if user is already a member
    if code in user_db[user_id]["leagues"]:
        return {"status": "success", "message": "You are already a member of this league."}
        
    # Add user to league
    league["members"].append({"telegram_id": user_id, "league_points": 0})
    
    # Add league to user's profile
    user_db[user_id]["leagues"][code] = 0
    
    return {
        "status": "success",
        "message": f"Successfully joined league '{league['name']}'.",
        "league_details": league
    }

@app.post("/league/my_leagues")
async def get_my_leagues(user_request: TelegramID):
    """
    Returns the list of leagues the user belongs to with rank and details.
    """
    user_id = user_request.telegram_id
    if user_id not in user_db:
        raise HTTPException(status_code=404, detail="User not found.")
        
    my_leagues_list = []
    
    for code in user_db[user_id]["leagues"].keys():
        if code in league_db:
            league = league_db[code]
            
            # 1. Calculate Rank (VERY IMPORTANT for the frontend display)
            # Sort members by points descending
            sorted_members = sorted(league["members"], key=lambda m: m["league_points"], reverse=True)
            
            # Find the user's index in the sorted list and add 1 for rank
            user_rank = next((i + 1 for i, m in enumerate(sorted_members) if m["telegram_id"] == user_id), "N/A")
            
            # 2. Extract needed details for the "My Leagues" display
            my_leagues_list.append({
                "league_avatar_url": league.get("avatar_url"),
                "league_name": league["name"],
                "league_description": league["description"],
                "user_rank": user_rank,
                "user_points": user_db[user_id]["leagues"][code],
                "member_count": len(league["members"]),
                "is_owner": league["owner_id"] == user_id, # Frontend uses this to display the "Owner" tag
                "code": code
            })
            
    return {
        "status": "success",
        "my_leagues": my_leagues_list
    }
    
@app.get("/league/discover")
async def discover_leagues():
    """
    Returns 3 random public leagues for the "Discover Leagues" section.
    """
    public_leagues = [
        l for l in league_db.values() if not l["is_private"]
    ]
    
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
            "member_count": len(league["members"]),
            "join_code": league["code"] # Needed for the 'Join' button
        })
        
    return {
        "status": "success",
        "public_leagues": discovery_list
    }

@app.post("/league/search")
async def search_leagues(search_data: LeagueSearch):
    """
    Allows users to search for public leagues by name.
    """
    query = search_data.query.strip().lower()
    
    # Filter for public leagues matching the query
    matching_leagues = [
        l for l in league_db.values() 
        if not l["is_private"] and query in l["name"].lower()
    ]
    
    # Format the data for the frontend display (similar to discover)
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
    
    # 2. STATIC FILES MOUNT (This is where the magic happens)
# This serves the index.html file from the 'static' directory when the user visits the root URL (/)
# It is placed AFTER CORS but BEFORE your API routes.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
