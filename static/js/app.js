// ============================================================
// Helpers
// ============================================================
function showMessage(elementId, message, type = "error") {
    const element = document.getElementById(elementId);
    if (!element) return;
    element.textContent = message;
    element.className = `form-message ${type}`;
}

function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {modal.classList.remove("hidden");}
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {modal.classList.add("hidden");}
}

function showComingSoon() {
    alert("هذه الميزة ستكون متاحة قريبًا 🌟");
}

// ============================================================
// Login
// ============================================================
const loginForm = document.getElementById("loginForm");

if (loginForm) {
    loginForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value.trim();

        if (!username || !password) {
            showMessage("loginMessage", "أدخل اسم المستخدم وكلمة المرور.");
            return;
        }

        try {
            const response = await fetch("/login", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({username, password})
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showMessage("loginMessage", data.message || "تعذر تسجيل الدخول.");
                return;
            }

            window.location.href = "/";

        } catch (error) {
            showMessage("loginMessage", "حدث خطأ في الاتصال. حاول مرة أخرى.");
        }
    });
}

// ============================================================
// Register
// ============================================================

const registerForm = document.getElementById("registerForm");

if (registerForm) {

    registerForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        const name = document.getElementById("name").value.trim();
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value.trim();

        if (!username || !password) {
            showMessage("registerMessage", "اسم المستخدم وكلمة المرور مطلوبة.");
            return;
        }

        try {
            const response = await fetch("/register", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    name,
                    username,
                    password
                })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showMessage("registerMessage", data.message || "تعذر إنشاء الحساب.");
                return;
            }

            window.location.href = "/";

        } catch (error) {
            showMessage("registerMessage", "حدث خطأ في الاتصال. حاول مرة أخرى.");
        }
    });
}

// ============================================================
// Feature Navigation
// ============================================================
function openFeature(feature) {
    if (feature === "story") {
        openModal("storyModal");
        return;
    }

    if (feature === "spelling") {
        showFeatureContent(
            "✏️",
            "مصحح الكتابة",
            `
                <p class="modal-description">
                    اكتب جملة باللغة العربية وسيساعدك أنيس
                    على اكتشاف الأخطاء وتحسين كتابتك.
                </p>

                <div class="form-group">
                    <label>اكتب جملتك</label>

                    <textarea
                        id="spellText"
                        placeholder="اكتب هنا..."
                    ></textarea>
                </div>

                <button
                    class="btn btn-primary btn-full"
                    onclick="checkSpelling()"
                >
                    تحقق من الكتابة
                </button>

                <div id="spellResult"></div>
            `
        );

        return;
    }

    if (feature === "game") {
        loadWordGame();
        return;
    }

    if (feature === "pronunciation") {
        loadPronunciation();
        return;
    }

    if (feature === "handwriting") {
        showFeatureContent(
            "✍️",
            "مقيم الخط",
            `
                <p>
                    اكتب الحرف المطلوب وارفع صورة كتابتك
                    ليقوم أنيس بتقييمها.
                </p>

                <div class="form-group">
                    <label>الحرف</label>

                    <input
                        id="handwritingLetter"
                        placeholder="مثال: ب"
                        maxlength="1"
                    >
                </div>

                <div class="form-group">
                    <label>صورة الكتابة</label>

                    <input
                        type="file"
                        id="handwritingImage"
                        accept="image/*"
                    >
                </div>

                <button
                    class="btn btn-primary btn-full"
                    onclick="evaluateHandwriting()"
                >
                    قيّم كتابتي
                </button>

                <div id="handwritingResult"></div>
            `
        );

        return;
    }


    if (feature === "reading") {
        showFeatureContent(
            "📖",
            "رفيق القراءة",
            `
                <p>
                    اقرأ القصص التي يصنعها أنيس،
                    واكتشف الكلمات الجديدة وأجب عن الأسئلة.
                </p>

                <button
                    class="btn btn-primary btn-full"
                    onclick="openFeature('story')"
                >
                    ابدأ القراءة
                </button>
            `
        );
    }
}

function showFeatureContent(icon, title, content) {
    const container = document.getElementById("featureContent");
    container.innerHTML = `
        <div class="modal-header">
            <div class="modal-icon">${icon}</div>
            <h2>${title}</h2>
        </div>
        ${content}
    `;
    openModal("featureModal");
}

// ============================================================
// Story
// ============================================================
async function generateStory() {
    const topic = document.getElementById("storyTopic").value;
    const character = document.getElementById("storyCharacter").value;
    const button = document.getElementById("generateStoryBtn");

    button.disabled = true;
    button.textContent = "أنيس يكتب قصتك...";

    try {
        const response = await fetch(
            "/api/story/generate",
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({topic, character})
            }
        );

        const data = await response.json();

        if (!response.ok || !data.success) {
            alert(data.message || "تعذر إنشاء القصة.");
            return;
        }

        displayStory(data);
        closeModal("storyModal");
        openModal("storyResultModal");

    } catch (error) {
        alert("حدث خطأ أثناء إنشاء القصة.");

    } finally {
        button.disabled = false;
        button.textContent = "✨ ابدأ القصة";
    }
}

// ============================================================
// Display Story
// ============================================================
function displayStory(data) {
    const storyContent = document.getElementById("storyContent");
    storyContent.textContent = data.story || "";

    const wordsContainer = document.getElementById("storyWords");
    const words = data.words?.words || [];

    if (words.length) {
        wordsContainer.innerHTML = `
            <h3>📚 كلمات جديدة</h3>
            <div>
                ${words.map(word => {
                    if (typeof word === "string") {
                        return `
                            <span class="word-chip">
                                ${word}
                            </span>
                        `;
                    }
                    return `
                        <span class="word-chip">
                            ${word.word || ""}
                        </span>
                    `;
                }).join("")}
            </div>
        `;

    } else {
        wordsContainer.innerHTML = "";
    }

    const questionsContainer = document.getElementById("storyQuestions");
    const questions = data.questions?.questions || [];

    if (questions.length) {
        questionsContainer.innerHTML = `
            <h3>🧠 أسئلة القصة</h3>
            ${questions.map((question, index) => {
                const text =
                    typeof question === "string"
                        ? question
                        : question.question || "";
                return `
                    <div class="question-card">
                        <strong>
                            ${index + 1}. ${text}
                        </strong>
                    </div>
                `;
            }).join("")}
        `;

    } else {
        questionsContainer.innerHTML = "";
    }
}

// ============================================================
// Spelling
// ============================================================
async function checkSpelling() {
    const text = document.getElementById("spellText").value.trim();

    if (!text) {return;}

    const result = document.getElementById("spellResult");
    result.innerHTML = `
        <div class="loading">
            أنيس يراجع كتابتك...
        </div>
    `;

    try {
        const response = await fetch(
            "/api/spell-check",
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text})
            }
        );

        const data = await response.json();
        if (!response.ok || !data.success) {
            result.innerHTML = `
                <p>
                    ${data.message || "حدث خطأ."}
                </p>
            `;
            return;
        }

        result.innerHTML = `
            <div class="result-card">
                <h3>✨ الملاحظات</h3>
                <p>${data.feedback || "كتابتك رائعة!"}</p>
            </div>
        `;

    } catch (error) {
        result.innerHTML = `
            <p>حدث خطأ في الاتصال.</p>
        `;
    }
}

// ============================================================
// Word Game
// ============================================================
async function loadWordGame() {
    try {
        const response = await fetch("/api/word-game");
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message);
        }

        const question = data.question;

        showFeatureContent(
            "🎮",
            "لعبة الكلمات",
            `
                <div class="question-card">
                    <p>${question.question}</p>
                    <h2>${question.word}</h2>
                </div>

                <div id="gameOptions">
                    ${question.options.map(option => `
                        <button
                            class="answer-option"
                            onclick="checkWordAnswer(
                                '${escapeHtml(question.word)}',
                                '${escapeHtml(option)}'
                            )"
                        >
                            ${option}
                        </button>
                    `).join("")}
                </div>
                <div id="gameResult"></div>
            `
        );

    } catch (error) {
        alert("تعذر تحميل سؤال اللعبة.");
    }
}

async function checkWordAnswer(word, answer) {
    const result = document.getElementById("gameResult");

    try {
        const response =
            await fetch(
                "/api/word-game/check",
                {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        word,
                        answer
                    })
                }
            );

        const data = await response.json();
        if (data.correct) {
            result.innerHTML = `
                <div class="success-message">أحسنت! 🎉</div>
            `;

        } else {
            result.innerHTML = `
                <div class="warning-message"> إجابة جيدة، حاول مرة أخرى 🌟</div>
            `;
        }

    } catch (error) {
        result.textContent ="حدث خطأ أثناء التحقق.";
    }

}

// ============================================================
// Pronunciation
// ============================================================
async function loadPronunciation() {
    try {
        const response = await fetch("/api/pronunciation/letters");
        const data = await response.json();
        const letters = data.letters || [];

        showFeatureContent(
            "🗣️",
            "محسن النطق",
            `
                <p>اختر حرفًا ثم سجّل نطقك له.</p>
                <div class="letter-grid">
                    ${letters.map(item => `
                        <button
                            class="letter-button"
                            onclick="selectPronunciationLetter(
                                '${item.letter}'
                            )"
                        >
                            ${item.letter}
                        </button>
                    `).join("")}
                </div>
                <div id="pronunciationArea"></div>
            `
        );

    } catch (error) {
        alert("تعذر تحميل الحروف.");
    }
}


function selectPronunciationLetter(letter) {
    const container = document.getElementById("pronunciationArea");

    container.innerHTML = `
        <div class="selected-letter">
            <span>الحرف المختار</span>
            <strong>${letter}</strong>
        </div>

        <div class="form-group">
            <label>تسجيل الصوت</label>
            <input type="file" id="pronunciationAudio" accept="audio/*">
        </div>

        <button
            class="btn btn-primary btn-full"
            onclick="evaluatePronunciation(
                '${letter}'
            )"
        >
            قيّم نطقي
        </button>
        <div id="pronunciationResult"></div>
    `;
}

async function evaluatePronunciation(letter) {
    const audio = document.getElementById("pronunciationAudio").files[0];

    if (!audio) {return;}
    const formData = new FormData();
    formData.append("target_letter", letter);
    formData.append("audio", audio);

    try {
        const response =
            await fetch(
                "/api/pronunciation/evaluate",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data = await response.json();
        const result = document.getElementById("pronunciationResult");

        result.innerHTML = `
            <div class="result-card">
                <h3>نتيجة النطق</h3>
                <pre>${JSON.stringify(
                    data.result,
                    null,
                    2
                )}</pre>
            </div>
        `;

    } catch (error) {
        alert("حدث خطأ أثناء تقييم النطق.");
    }
}

// ============================================================
// Handwriting
// ============================================================
async function evaluateHandwriting() {
    const letter = document.getElementById("handwritingLetter").value.trim();
    const image = document.getElementById("handwritingImage").files[0];

    if (!letter || !image) {return;}

    const formData = new FormData();
    formData.append("target_letter", letter);
    formData.append("image", image);

    try {
        const response =
            await fetch(
                "/api/handwriting/evaluate",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data = await response.json();
        const result = document.getElementById("handwritingResult");

        if (!response.ok || !data.success) {
            result.innerHTML = `
                <p>${data.message || "حدث خطأ."}</p>
            `;
            return;
        }

        const evaluation = data.result;

        result.innerHTML = `
            <div class="result-card">
                <h3>نتيجة كتابتك ✨</h3>
                <div class="score">${evaluation.score}</div>
                <p>${evaluation.grade}</p>
                <p>
                    الحرف المتوقع:
                    ${evaluation.predicted_letter}
                </p>
            </div>
        `;

    } catch (error) {
        alert("حدث خطأ أثناء تقييم الكتابة.");
    }
}

// ============================================================
// Utility
// ============================================================
function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}