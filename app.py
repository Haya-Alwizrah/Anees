import os
import json
import random
import tempfile
import subprocess
from pathlib import Path
from functools import wraps
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

from flask import Flask, request, jsonify, session, redirect, url_for, render_template, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

from src.rag import StoryGenerator
from src.spell_checker import SpellChecker
from src.pronunciation import PronunciationEvaluator

# -------------------------------------------------------------[ Paths ]--------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SRC_DIR = BASE_DIR / "src"
DATASETS_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "model"

STORIES_DATASET = DATASETS_DIR / "Formatted-MSA-prompts-stories-for-fine-tuning.csv"
QUESTIONS_FILE = DATASETS_DIR / "questions.json"
REFERENCE_AUDIO_DIR = DATASETS_DIR / "reference_audio"
HANDWRITING_MODEL = MODEL_DIR / "arabic_handwriting_model.h5"
HANDWRITING_EVALUATOR = BASE_DIR / "notebooks" / "Handwriting_Enhancer" / "evaluator.py"
USERS_FILE = DATASETS_DIR / "users.json"

# -------------------------------------------------------------[ Flask ]-------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")

# -------------------------------------------------------------[ Configuration ]-------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_KEY2")
print(GROQ_API_KEY)
STORY_MODEL = os.getenv("STORY_MODEL", "openai/gpt-oss-120b")
SPELL_MODEL = os.getenv("SPELL_MODEL", "openai/gpt-oss-120b")
PRONUNCIATION_THRESHOLD = float(os.getenv("PRONUNCIATION_THRESHOLD", "75"))

# -------------------------------------------------------------[ Runtime AI objects ]------------------------------------------------------------
story_generator = None
spell_checker = None
pronunciation_evaluator = None
handwriting_model = None
handwriting_evaluator = None

# ------------------------------------------------------------- Users -------------------------------------------------------------
def login_required():
    return "username" in session

def require_login(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not login_required():
            return jsonify({
                "success": False,
                "message": "يجب تسجيل الدخول."
            }), 401
        return func(*args, **kwargs)
    
    return wrapper

def load_users():
    if not USERS_FILE.exists():
        return {}

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def get_current_user():
    username = session.get("username")

    if not username:
        return None

    users = load_users()
    return users.get(username)

def update_progress(feature):
    username = session.get("username")

    if not username: return

    users = load_users()

    if username not in users: return
    if feature not in users[username]["progress"]: return

    users[username]["progress"][feature] += 1
    save_users(users)

# ------------------------------------------------------------- Authentication -------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    data = request.get_json(silent=True) or request.form

    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    name = str(data.get("name", "")).strip()

    if not username or not password:
        return jsonify({
            "success": False,
            "message": "اسم المستخدم وكلمة المرور مطلوبة."
        }), 400

    users = load_users()

    if username in users:
        return jsonify({
            "success": False,
            "message": "اسم المستخدم موجود مسبقًا."
        }), 409

    users[username] = {
        "username": username,
        "name": name or username,
        "password": generate_password_hash(password),
        "profile": {"age": None, "level": None},
        "progress": {
            "stories": 0,
            "spelling": 0,
            "word_game": 0,
            "pronunciation": 0,
            "handwriting": 0,
        },
    }

    save_users(users)
    session["username"] = username

    return jsonify({
        "success": True,
        "message": "تم إنشاء الحساب بنجاح.",
        "user": users[username],
    })


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    users = load_users()
    user = users.get(username)

    if not user:
        return jsonify({
            "success": False,
            "message": "اسم المستخدم أو كلمة المرور غير صحيحة."
        }), 401

    if not check_password_hash(user.get("password", ""), password):
        return jsonify({
            "success": False,
            "message": "اسم المستخدم أو كلمة المرور غير صحيحة."
        }), 401

    session["username"] = username
    return jsonify({
        "success": True,
        "message": "تم تسجيل الدخول.",
        "user": user,
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ------------------------------------------------------------- Profile -------------------------------------------------------------
@app.route("/profile", methods=["GET", "POST"])
@require_login
def profile():
    username = session["username"]
    users = load_users()
    user = users.get(username)

    if not user:
        session.clear()
        return jsonify({
            "success": False,
            "message": "المستخدم غير موجود."
        }), 404

    # GET profile
    if request.method == "GET":
        return jsonify({
            "success": True,
            "user": user,
        })

    # UPDATE profile
    data = request.get_json(silent=True) or request.form
    if "name" in data: user["name"] = str(data["name"]).strip()
    if "age" in data: user["profile"]["age"] = data["age"]
    if "level" in data: user["profile"]["level"] = data["level"]

    users[username] = user
    save_users(users)

    return jsonify({
        "success": True,
        "user": user,
    })

# Home
@app.route("/")
def home():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("index.html", user=get_current_user())

@app.route("/story")
@require_login
def story_page():
    return render_template("story.html", user=get_current_user())


@app.route("/spelling")
@require_login
def spelling_page():
    return render_template("spelling.html", user=get_current_user())


@app.route("/word-game")
@require_login
def word_game_page():
    return render_template("word_game.html", user=get_current_user())

@app.route("/pronunciation")
@require_login
def pronunciation_page():
    return render_template("pronunciation.html", user=get_current_user())

@app.route("/handwriting")
@require_login
def handwriting_page():
    return render_template("handwriting.html", user=get_current_user())

@app.route("/library")
@require_login
def library_page():
    return render_template("library.html", user=get_current_user())

@app.route("/achievements")
@require_login
def achievements_page():
    return render_template("achievements.html", user=get_current_user())

# =============================================================================
# 1. STORY GENERATION
# =============================================================================
def get_story_generator():
    global story_generator

    if story_generator is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        story_generator = StoryGenerator(
            data_path=str(STORIES_DATASET),
            api_key=GROQ_API_KEY,
            model_name=STORY_MODEL,
        )

    return story_generator

@app.route("/api/story/generate", methods=["POST"])
@require_login
def generate_story():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "الصدق")
    character = data.get("character", "أرنب")

    try:
        generator = get_story_generator()

        # Generate Story
        story = generator.generate_story(
            search_t=5,
            topic=topic,
            character=character,
            n_results=5,
        )

        # Difficult Words
        words_raw = generator.explain_words(story)

        try:
            words = json.loads(words_raw)
        except (json.JSONDecodeError, TypeError):
            words = {"words": []}

        # Questions
        questions_raw = generator.generate_QA(story)

        try:
            questions = json.loads(questions_raw)
        except (json.JSONDecodeError, TypeError):
            questions = {"questions": []}

        # Update Progress
        update_progress("stories")
        
        # Response
        return jsonify({
            "success": True,
            "story": story,
            "words": words,
            "questions": questions,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 500


# =============================================================================
# 2. SPELLING & GRAMMAR CHECKER
# =============================================================================
def get_spell_checker():
    global spell_checker

    if spell_checker is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        spell_checker = SpellChecker(
            api_key=GROQ_API_KEY,
            model_name=SPELL_MODEL,
        )

    return spell_checker

@app.route("/api/spell-check", methods=["POST"])
@require_login
def spell_check():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "success": False,
            "message": "النص فارغ."
        }), 400

    try:
        checker = get_spell_checker()
        feedback, errors = checker.correct(text)

        # Update Progress
        update_progress("spelling")

        # Response
        return jsonify({
            "success": True,
            "text": text,
            "feedback": feedback,
            "errors": errors,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 500


# =============================================================================
# 3. WORD GAME
# =============================================================================
def load_word_questions():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")

    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid questions.json: {e}")

    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")

    return questions


@app.route("/api/word-game", methods=["GET"])
@require_login
def word_game_question():
    try:
        questions = load_word_questions()

        if not questions:
            return jsonify({
                "success": False,
                "message": "لا توجد أسئلة."
            }), 404

        # Select a random question
        question = random.choice(questions)

        return jsonify({
            "success": True,
            "question": {
                "word": question["word"],
                "question": question["question"],
                "options": question["options"],
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 500


@app.route("/api/word-game/check", methods=["POST"])
@require_login
def check_word_game():
    data = request.get_json(silent=True) or {}
    word = str(data.get("word", "")).strip()
    selected_answer = str(data.get("answer", "")).strip()

    if not word or not selected_answer:
        return jsonify({
            "success": False,
            "message": "الكلمة والإجابة مطلوبة."
        }), 400

    try:
        questions = load_word_questions()

        # Find the original question
        question = next((q for q in questions if q.get("word", "").strip() == word), None)

        if question is None:
            return jsonify({
                "success": False,
                "message": "السؤال غير موجود."
            }), 404

        correct_answer = str(question["answer"]).strip()
        is_correct = (selected_answer == correct_answer)

        # Update Progress
        if is_correct:
            update_progress("word_game")

        # Response
        return jsonify({
            "success": True,
            "correct": is_correct,
            "correct_answer": correct_answer if is_correct else None
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 500


# =============================================================================
# 4. Pronunciation Evaluator
# =============================================================================
def get_pronunciation_evaluator():
    global pronunciation_evaluator

    if pronunciation_evaluator is None:
        pronunciation_evaluator = PronunciationEvaluator(threshold=PRONUNCIATION_THRESHOLD)

    return pronunciation_evaluator

@app.route("/api/pronunciation/letters", methods=["GET"])
@require_login
def pronunciation_letters():
    if not REFERENCE_AUDIO_DIR.exists():
        return jsonify({
            "success": True,
            "letters": [],
        })

    letters = []
    for path in sorted(REFERENCE_AUDIO_DIR.glob("*.wav")):
        letters.append({
            "letter": path.stem,
            "audio": f"/api/pronunciation/reference/{path.name}"
        })

    return jsonify({
        "success": True,
        "letters": letters,
    })


@app.route("/api/pronunciation/reference/<filename>", methods=["GET"])
@require_login
def pronunciation_reference(filename):
    return send_from_directory(REFERENCE_AUDIO_DIR, filename)

@app.route("/api/pronunciation/evaluate", methods=["POST"])
@require_login
def pronunciation_evaluate():
    target_letter = request.form.get("target_letter")
    audio = request.files.get("audio")

    if not target_letter:
        return jsonify({
            "success": False,
            "message": "الحرف المطلوب غير موجود."
        }), 400

    if not audio:
        return jsonify({
            "success": False,
            "message": "ملف الصوت غير موجود."
        }), 400

    if audio.filename == "":
        return jsonify({
            "success": False,
            "message": "ملف التسجيل فارغ."
        }), 400

    input_path = None
    wav_path = None

    try:
        # 1. Save browser recording
        original_suffix = Path(audio.filename or ".webm").suffix.lower()
        if not original_suffix:
            original_suffix = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as temp:
            audio.save(temp.name)
            input_path = temp.name

        # Check that file actually contains data
        if not os.path.exists(input_path):
            raise RuntimeError("تعذر حفظ التسجيل.")

        file_size = os.path.getsize(input_path)
        print(f"[Pronunciation] Uploaded file: {input_path} ({file_size} bytes)")
        if file_size == 0:
            raise ValueError("التسجيل فارغ. حاول التسجيل مرة أخرى.")

        # 2. Convert WebM/Opus -> WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            wav_path = temp_wav.name

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i", 
            input_path,
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            wav_path
        ]

        print("[Pronunciation] Converting audio to WAV...")

        process = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            print("[Pronunciation] FFmpeg error:")
            print(process.stderr)
            raise RuntimeError("تعذر تحويل التسجيل الصوتي.")

        # 3. Make sure WAV exists
        if not os.path.exists(wav_path):
            raise RuntimeError("لم يتم إنشاء ملف الصوت بعد التحويل.")

        wav_size = os.path.getsize(wav_path)

        print(f"[Pronunciation] WAV created: {wav_path} ({wav_size} bytes)")

        if wav_size == 0:
            raise ValueError("ملف الصوت الناتج فارغ.")

        # 4. Evaluate pronunciation
        evaluator = get_pronunciation_evaluator()
        result = evaluator.evaluate(audio_path=wav_path, target_letter=target_letter)

        # 5. Update progress
        update_progress("pronunciation")
        return jsonify({
            "success": True,
            "result": result,
        })

    except Exception as e:
        print("[Pronunciation] Evaluation error:", str(e))
        return jsonify({
            "success": False,
            "message": str(e),
        }), 500

    finally:
        # Delete original browser recording
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

        # Delete converted WAV
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

# =============================================================================
# 5. HANDWRITING
# =============================================================================
handwriting_model = None
handwriting_evaluator = None

def load_handwriting_model():
    global handwriting_model

    if handwriting_model is not None:
        return handwriting_model

    if not HANDWRITING_MODEL.exists():
        raise FileNotFoundError(f"Handwriting model not found")

    try:
        import tensorflow as tf
        import keras
        
        print(f"[Handwriting] TensorFlow: {tf.__version__}")
        print(f"[Handwriting] Keras: {keras.__version__}")

        original_glorot = keras.initializers.GlorotUniform

        class CompatibleGlorotUniform(original_glorot):
            def __init__(
                self,
                seed=None,
                input_axes=None,
                output_axes=None,
                **kwargs
            ):
                super().__init__(seed=seed, **kwargs)

        handwriting_model = tf.keras.models.load_model(
            HANDWRITING_MODEL,
            compile=False,
            custom_objects={
                "GlorotUniform": CompatibleGlorotUniform
            }
        )

        print("[Handwriting] Model loaded successfully.")

        return handwriting_model

    except Exception as e:
        raise RuntimeError(f"Could not load handwriting model: {e}")

def load_handwriting_evaluator():
    global handwriting_evaluator

    if handwriting_evaluator is not None:
        return handwriting_evaluator

    if not HANDWRITING_EVALUATOR.exists():
        raise FileNotFoundError(f"Handwriting evaluator not found")

    import sys
    evaluator_dir = str(HANDWRITING_EVALUATOR.parent)

    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)

    import evaluator
    handwriting_evaluator = evaluator
    return handwriting_evaluator


@app.route("/api/handwriting/evaluate", methods=["POST"])
@require_login
def handwriting_evaluate():
    image = request.files.get("image")
    target_letter = request.form.get("target_letter")
    adaptive = (request.form.get("adaptive", "false").lower() == "true")

    if not image:
        return jsonify({
            "success": False,
            "message": "صورة الكتابة غير موجودة."
        }), 400

    if not target_letter:
        return jsonify({
            "success": False,
            "message": "الحرف المطلوب غير موجود."
        }), 400

    temp_path = None

    try:
        suffix = Path(image.filename or ".png").suffix.lower()

        if suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
            suffix = ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            image.save(temp.name)
            temp_path = temp.name

        evaluator = load_handwriting_evaluator()
        model = load_handwriting_model()

        letters = evaluator.ARABIC_LETTERS

        if target_letter not in letters:
            return jsonify({
                "success": False,
                "message": "الحرف غير مدعوم."
            }), 400

        target_index = (letters.index(target_letter) + 1)

        result = evaluator.evaluate_handwriting(
            model=model,
            image_path=temp_path,
            target_letter_index=target_index,
            adaptive=adaptive,
            verbose=False,
            preview=False,
        )

        # Convert result to JSON-safe values
        response = {
            "predicted_letter": result["predicted_letter"],
            "predicted_name": result["predicted_name"],
            "target_letter": result["target_letter"],
            "score": round(float(result["score"]), 2),
            "grade": result["grade"],
            "tier": result["tier"],
            "correct": bool( result["correct"]),
            "top3": [
                {
                    "letter": letter,
                    "confidence":round(float(score), 2),
                }
                for _, letter, score in result["top3"]
            ],
        }

        # Update Progress
        update_progress("handwriting")

        return jsonify({
            "success": True,
            "result": response,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 500

    finally:
        if (temp_path and os.path.exists(temp_path)):
            os.remove(temp_path)

# ------------------------------------------------------------- System Status -------------------------------------------------------------
@app.route("/api/status")
def status():
    return jsonify({
        "app": "Anees",
        "status": "running",
        "features": {
            "story": STORIES_DATASET.exists(),
            "spell_checker": bool(GROQ_API_KEY),
            "word_game": QUESTIONS_FILE.exists(),
            "pronunciation": PronunciationEvaluator is not None,
            "handwriting": HANDWRITING_MODEL.exists(),
        },

        "paths": {
            "stories_dataset": STORIES_DATASET.exists(),
            "questions": QUESTIONS_FILE.exists(),
            "reference_audio": REFERENCE_AUDIO_DIR.exists(),
            "handwriting_model": HANDWRITING_MODEL.exists(),
            "handwriting_evaluator": HANDWRITING_EVALUATOR.exists(),
        },
    })

# ------------------------------------------------------------- Error Handling -------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "الصفحة غير موجودة."
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "حدث خطأ داخلي في التطبيق."
    }), 500

# ------------------------------------------------------------- Run -------------------------------------------------------------
def load_all_models():
    global story_generator
    global spell_checker
    global pronunciation_evaluator
    global handwriting_model
    global handwriting_evaluator

    print("\n" + "=" * 60)
    print("Loading all AI models...")
    print("=" * 60)

    # Story Generator
    print("\n[1/4] Loading Story Generator...")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    story_generator = StoryGenerator(
        data_path=str(STORIES_DATASET),
        api_key=GROQ_API_KEY,
        model_name=STORY_MODEL,
    )

    story_generator.prepare()
    print("Loading reranker...")
    story_generator._rerank("اختبار", ["هذه قصة اختبار"])
    print("[1/4] Story Generator READY")

    # Spell Checker
    print("\n[2/4] Loading Spell Checker...")
    spell_checker = SpellChecker(
        api_key=GROQ_API_KEY,
        model_name=SPELL_MODEL,
    )

    print("[2/4] Spell Checker READY")

    # Pronunciation
    print("\n[3/4] Loading Pronunciation Model...")
    pronunciation_evaluator = PronunciationEvaluator(threshold=PRONUNCIATION_THRESHOLD)
    print("[3/4] Pronunciation Model READY")

    # Handwriting
    print("\n[4/4] Loading Handwriting Model...")
    handwriting_evaluator = load_handwriting_evaluator()
    handwriting_model = load_handwriting_model()

    print("[4/4] Handwriting Model READY")

    print("\n" + "=" * 60)
    print("ALL AI MODELS ARE LOADED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    load_all_models()

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=(os.getenv("FLASK_DEBUG", "false").lower() == "true"),
    )