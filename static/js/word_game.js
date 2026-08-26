// ============================================================
// WORD GAME
// ============================================================

const TOTAL_QUESTIONS = 10;

// Current game state
let currentQuestionIndex = 0;
let score = 0;
let currentQuestion = null;

// Keep track of words already used in this round
let usedWords = new Set();


// ============================================================
// DOM ELEMENTS
// ============================================================

const gameLoading = document.getElementById("gameLoading");
const questionCard = document.getElementById("questionCard");
const gameResult = document.getElementById("gameResult");
const gameError = document.getElementById("gameError");

const questionCounter = document.getElementById("questionCounter");
const progressFill = document.getElementById("progressFill");

const questionText = document.getElementById("questionText");
const optionsContainer = document.getElementById("optionsContainer");

const answerFeedback = document.getElementById("answerFeedback");
const feedbackIcon = document.getElementById("feedbackIcon");
const feedbackTitle = document.getElementById("feedbackTitle");
const feedbackText = document.getElementById("feedbackText");

const nextButton = document.getElementById("nextButton");

const scoreValue = document.getElementById("scoreValue");
const scoreProgressFill = document.getElementById("scoreProgressFill");
const resultMessage = document.getElementById("resultMessage");

const errorMessage = document.getElementById("errorMessage");

const restartButton = document.getElementById("restartButton");
const retryButton = document.getElementById("retryButton");


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    startGame();
});


// ============================================================
// START GAME
// ============================================================

function startGame() {

    currentQuestionIndex = 0;
    score = 0;
    currentQuestion = null;

    usedWords = new Set();

    hideElement(gameResult);
    hideElement(gameError);
    showElement(gameLoading);
    hideElement(questionCard);

    updateProgress();

    loadQuestion();
}


// ============================================================
// LOAD RANDOM QUESTION
// ============================================================

async function loadQuestion() {

    showElement(gameLoading);
    hideElement(questionCard);
    hideElement(gameError);

    try {

        let question = null;

        // Try several times to avoid duplicate questions
        // inside the same round.
        for (let attempt = 0; attempt < 10; attempt++) {

            const response = await fetch("/api/word-game", {
                method: "GET",
                credentials: "same-origin"
            });

            if (!response.ok) {
                throw new Error("تعذر تحميل السؤال.");
            }

            const data = await response.json();

            if (!data.success || !data.question) {
                throw new Error(
                    data.message || "تعذر تحميل السؤال."
                );
            }

            const candidate = data.question;

            if (!usedWords.has(candidate.word)) {
                question = candidate;
                break;
            }
        }

        // If there are not enough unique questions,
        // allow a repeated question.
        if (!question) {

            const response = await fetch("/api/word-game", {
                method: "GET",
                credentials: "same-origin"
            });

            if (!response.ok) {
                throw new Error("تعذر تحميل السؤال.");
            }

            const data = await response.json();

            if (!data.success || !data.question) {
                throw new Error(
                    data.message || "تعذر تحميل السؤال."
                );
            }

            question = data.question;
        }

        currentQuestion = question;

        usedWords.add(question.word);

        displayQuestion(question);

    } catch (error) {

        console.error("Word game error:", error);

        showError(
            error.message || "تعذر تحميل السؤال. حاول مرة أخرى."
        );
    }
}


// ============================================================
// DISPLAY QUESTION
// ============================================================

function displayQuestion(question) {

    hideElement(gameLoading);
    hideElement(gameError);

    showElement(questionCard);

    // Counter
    questionCounter.textContent =
        `السؤال ${currentQuestionIndex + 1} من ${TOTAL_QUESTIONS}`;

    // Progress
    updateProgress();

    // Question
    questionText.textContent = question.question;

    // Clear previous options
    optionsContainer.innerHTML = "";

    // Reset feedback
    hideElement(answerFeedback);
    hideElement(nextButton);

    // Create options
    question.options.forEach((option, index) => {

        const button = document.createElement("button");

        button.type = "button";
        button.className = "game-option";

        button.dataset.answer = option;

        // Number
        const number = document.createElement("span");
        number.className = "option-number";
        number.textContent = index + 1;

        // Text
        const text = document.createElement("span");
        text.className = "option-text";
        text.textContent = option;

        // Arrow/check area
        const icon = document.createElement("span");
        icon.className = "option-icon";
        icon.textContent = "›";

        button.appendChild(number);
        button.appendChild(text);
        button.appendChild(icon);

        button.addEventListener("click", () => {
            checkAnswer(option, button);
        });

        optionsContainer.appendChild(button);
    });
}


// ============================================================
// CHECK ANSWER
// ============================================================

async function checkAnswer(selectedAnswer, selectedButton) {

    // Prevent clicking multiple answers
    disableOptions();

    try {

        const response = await fetch("/api/word-game/check", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            credentials: "same-origin",

            body: JSON.stringify({
                word: currentQuestion.word,
                answer: selectedAnswer
            })
        });

        if (!response.ok) {
            throw new Error("تعذر التحقق من الإجابة.");
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(
                data.message || "تعذر التحقق من الإجابة."
            );
        }

        if (data.correct) {

            score++;

            selectedButton.classList.add("correct");

            showFeedback(
                true,
                "أحسنت! 🌟",
                "إجابة صحيحة، استمر!"
            );

        } else {

            selectedButton.classList.add("wrong");

            // Find correct answer
            const correctButton = Array.from(
                optionsContainer.querySelectorAll(".game-option")
            ).find(button =>
                button.dataset.answer === currentQuestion.word
            );

            if (correctButton) {
                correctButton.classList.add("correct");
            }

            showFeedback(
                false,
                "محاولة جميلة!",
                `الإجابة الصحيحة هي: ${currentQuestion.word}`
            );
        }

        // Show next button
        showElement(nextButton);

        // If this is the last question
        if (currentQuestionIndex === TOTAL_QUESTIONS - 1) {

            nextButton.textContent = "عرض النتيجة";

        } else {

            nextButton.textContent = "السؤال التالي";
        }

    } catch (error) {

        console.error("Answer check error:", error);

        enableOptions();

        showError(
            error.message || "حدث خطأ أثناء التحقق من الإجابة."
        );
    }
}


// ============================================================
// FEEDBACK
// ============================================================

function showFeedback(correct, title, message) {

    showElement(answerFeedback);

    feedbackTitle.textContent = title;
    feedbackText.textContent = message;

    answerFeedback.classList.remove(
        "feedback-correct",
        "feedback-wrong"
    );

    if (correct) {

        answerFeedback.classList.add("feedback-correct");

        feedbackIcon.textContent = "✓";

    } else {

        answerFeedback.classList.add("feedback-wrong");

        feedbackIcon.textContent = "!";
    }
}


// ============================================================
// NEXT QUESTION
// ============================================================

nextButton.addEventListener("click", () => {

    if (currentQuestionIndex >= TOTAL_QUESTIONS - 1) {

        finishGame();

        return;
    }

    currentQuestionIndex++;

    loadQuestion();
});


// ============================================================
// FINISH GAME
// ============================================================

function finishGame() {

    hideElement(questionCard);
    hideElement(gameLoading);
    hideElement(gameError);

    showElement(gameResult);

    // Score
    scoreValue.textContent = score;

    // Score percentage
    const percentage =
        (score / TOTAL_QUESTIONS) * 100;

    scoreProgressFill.style.width =
        `${percentage}%`;

    // Result message
    if (score === 10) {

        resultMessage.textContent =
            "ممتاز جدًا! أجبت عن جميع الأسئلة بشكل صحيح! 🌟";

    } else if (score >= 8) {

        resultMessage.textContent =
            "رائع جدًا! لديك معرفة جميلة بالكلمات! 👏";

    } else if (score >= 5) {

        resultMessage.textContent =
            "أداء جميل! استمر في التعلم وستصبح أفضل! 💪";

    } else {

        resultMessage.textContent =
            "محاولة جميلة! لا بأس، جرّب مرة أخرى وتعلّم كلمات جديدة! 🌱";
    }

    // Update progress bar to 100%
    progressFill.style.width = "100%";

    questionCounter.textContent =
        `أكملت ${TOTAL_QUESTIONS} أسئلة`;
}


// ============================================================
// RESTART
// ============================================================

restartButton.addEventListener("click", () => {
    startGame();
});


// ============================================================
// RETRY
// ============================================================

retryButton.addEventListener("click", () => {

    hideElement(gameError);
    showElement(gameLoading);

    loadQuestion();
});


// ============================================================
// UPDATE PROGRESS
// ============================================================

function updateProgress() {

    const progress =
        ((currentQuestionIndex) / TOTAL_QUESTIONS) * 100;

    progressFill.style.width =
        `${progress}%`;
}


// ============================================================
// OPTIONS
// ============================================================

function disableOptions() {

    const buttons =
        optionsContainer.querySelectorAll(".game-option");

    buttons.forEach(button => {
        button.disabled = true;
    });
}


function enableOptions() {

    const buttons =
        optionsContainer.querySelectorAll(".game-option");

    buttons.forEach(button => {
        button.disabled = false;
    });
}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    hideElement(gameLoading);
    hideElement(questionCard);
    hideElement(gameResult);

    showElement(gameError);

    errorMessage.textContent = message;
}


// ============================================================
// HELPERS
// ============================================================

function showElement(element) {

    if (element) {
        element.classList.remove("hidden");
    }
}


function hideElement(element) {

    if (element) {
        element.classList.add("hidden");
    }
}