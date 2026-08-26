document.addEventListener("DOMContentLoaded", () => {

    const diaryText = document.getElementById("diaryText");
    const charCount = document.getElementById("charCount");

    const checkBtn = document.getElementById("checkSpellingBtn");
    const checkBtnText = document.getElementById("checkBtnText");
    const checkLoader = document.getElementById("checkLoader");

    const spellingError = document.getElementById("spellingError");

    const loadingSection = document.getElementById("spellingLoading");
    const resultSection = document.getElementById("spellingResult");

    const feedbackText = document.getElementById("feedbackText");
    const originalText = document.getElementById("originalText");

    const errorsSection = document.getElementById("errorsSection");
    const errorsContainer = document.getElementById("errorsContainer");

    const perfectResult = document.getElementById("perfectResult");

    const newDiaryBtn = document.getElementById("newDiaryBtn");


    // =========================================================
    // Character Counter
    // =========================================================

    diaryText.addEventListener("input", () => {

        const length = diaryText.value.length;

        charCount.textContent = `${length} / 3000`;

    });


    // =========================================================
    // Check Spelling
    // =========================================================

    checkBtn.addEventListener("click", checkDiary);


    async function checkDiary() {

        const text = diaryText.value.trim();

        // Empty text
        if (!text) {

            showError("اكتب شيئًا من يومياتك أولًا 🌱");

            diaryText.focus();

            return;
        }


        // Minimum text
        if (text.length < 3) {

            showError("اكتب جملة أو أكثر حتى أستطيع مساعدتك ✨");

            diaryText.focus();

            return;
        }


        hideError();

        setLoading(true);

        hideResults();


        try {

            const response = await fetch("/api/spell-check", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })

            });


            const data = await response.json();


            if (!response.ok || !data.success) {

                throw new Error(
                    data.message || "حدث خطأ أثناء مراجعة النص."
                );

            }


            displayResults(data);


        } catch (error) {

            console.error("Spell check error:", error);

            showError(
                error.message ||
                "حدث خطأ أثناء مراجعة يومياتك. حاول مرة أخرى."
            );

        } finally {

            setLoading(false);

        }

    }


    // =========================================================
    // Display Results
    // =========================================================

    function displayResults(data) {

        const text = data.text || "";
        const feedback = data.feedback || "";

        const errors = Array.isArray(data.errors)
            ? data.errors
            : [];


        // Original text
        originalText.textContent = text;


        // Feedback
        feedbackText.textContent =
            feedback ||
            "أحسنت! استمر في الكتابة والمحاولة، وستتحسن أكثر.";


        // Clear old errors
        errorsContainer.innerHTML = "";


        // No errors
        if (errors.length === 0) {

            errorsSection.classList.add("hidden");

            perfectResult.classList.remove("hidden");

        } else {

            errorsSection.classList.remove("hidden");

            perfectResult.classList.add("hidden");

            renderErrors(errors);

        }


        resultSection.classList.remove("hidden");


        // Scroll to result
        setTimeout(() => {

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }, 100);

    }


    // =========================================================
    // Render Errors
    // =========================================================

    function renderErrors(errors) {

        errors.forEach((error, index) => {

            const card = document.createElement("div");

            card.className = "spelling-error-card";


            const number = document.createElement("div");

            number.className = "error-number";

            number.textContent = index + 1;


            const content = document.createElement("div");

            content.className = "error-content";


            const words = document.createElement("div");

            words.className = "error-words";


            const wrong = document.createElement("span");

            wrong.className = "wrong-word";

            wrong.textContent =
                getErrorValue(
                    error,
                    ["wrong", "incorrect", "word", "original"]
                );


            const arrow = document.createElement("span");

            arrow.className = "error-arrow";

            arrow.textContent = "←";


            const correct = document.createElement("span");

            correct.className = "correct-word";

            correct.textContent =
                getErrorValue(
                    error,
                    ["correct", "correction", "fixed"]
                );


            words.appendChild(wrong);
            words.appendChild(arrow);
            words.appendChild(correct);


            content.appendChild(words);


            const reason = getErrorValue(
                error,
                ["reason", "explanation", "feedback", "description"]
            );


            if (reason) {

                const reasonElement =
                    document.createElement("p");

                reasonElement.className = "error-reason";

                reasonElement.textContent = reason;

                content.appendChild(reasonElement);

            }


            card.appendChild(number);
            card.appendChild(content);

            errorsContainer.appendChild(card);

        });

    }


    // =========================================================
    // Get Error Value
    // =========================================================

    function getErrorValue(error, keys) {

        if (typeof error === "string") {
            return error;
        }


        if (!error || typeof error !== "object") {
            return "";
        }


        for (const key of keys) {

            if (
                error[key] !== undefined &&
                error[key] !== null
            ) {

                return String(error[key]);

            }

        }


        return "";

    }


    // =========================================================
    // Loading
    // =========================================================

    function setLoading(isLoading) {

        checkBtn.disabled = isLoading;


        if (isLoading) {

            checkBtnText.textContent =
                "جاري المراجعة...";

            checkLoader.classList.remove("hidden");

            loadingSection.classList.remove("hidden");

        } else {

            checkBtnText.textContent =
                "راجع يومياتي";

            checkLoader.classList.add("hidden");

            loadingSection.classList.add("hidden");

        }

    }


    // =========================================================
    // Error Message
    // =========================================================

    function showError(message) {

        spellingError.textContent = message;

        spellingError.classList.remove("hidden");

    }


    function hideError() {

        spellingError.textContent = "";

        spellingError.classList.add("hidden");

    }


    // =========================================================
    // Hide Results
    // =========================================================

    function hideResults() {

        resultSection.classList.add("hidden");

        perfectResult.classList.add("hidden");

        errorsSection.classList.remove("hidden");

        errorsContainer.innerHTML = "";

    }


    // =========================================================
    // New Diary
    // =========================================================

    newDiaryBtn.addEventListener("click", () => {

        diaryText.value = "";

        charCount.textContent = "0 / 3000";

        hideResults();

        hideError();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

        setTimeout(() => {

            diaryText.focus();

        }, 400);

    });

});