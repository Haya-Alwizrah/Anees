document.addEventListener("DOMContentLoaded", () => {

    const canvas = document.getElementById("handwritingCanvas");
    const ctx = canvas.getContext("2d");

    const placeholder = document.getElementById("canvasPlaceholder");

    const clearBtn = document.getElementById("clearBtn");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const retryBtn = document.getElementById("retryBtn");

    const analyzeText = document.getElementById("analyzeText");
    const analyzeLoader = document.getElementById("analyzeLoader");

    const errorBox = document.getElementById("handwritingError");

    const resultSection = document.getElementById("handwritingResult");
    const scoreValue = document.getElementById("scoreValue");
    const resultTitle = document.getElementById("resultTitle");
    const resultMessage = document.getElementById("resultMessage");
    const feedbackText = document.getElementById("feedbackText");
    const resultIcon = document.getElementById("resultIcon");

    let isDrawing = false;
    let hasDrawing = false;

    let lastX = 0;
    let lastY = 0;


    /* =====================================================
       CANVAS SETUP
    ===================================================== */

    function setupCanvas() {

        const rect = canvas.getBoundingClientRect();

        const dpr = window.devicePixelRatio || 1;

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        ctx.scale(dpr, dpr);

        ctx.lineWidth = 5;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        ctx.strokeStyle = "#3A3335";

        clearCanvas();
    }


    /* =====================================================
       CLEAR CANVAS
    ===================================================== */

    function clearCanvas() {

        const rect = canvas.getBoundingClientRect();

        ctx.clearRect(
            0,
            0,
            rect.width,
            rect.height
        );

        hasDrawing = false;

        placeholder.classList.remove("hidden");

        hideError();

        resultSection.classList.add("hidden");
    }


    /* =====================================================
       GET POINTER POSITION
    ===================================================== */

    function getPointerPosition(event) {

        const rect = canvas.getBoundingClientRect();

        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }


    /* =====================================================
       START DRAWING
    ===================================================== */

    function startDrawing(event) {

        event.preventDefault();

        isDrawing = true;

        const position = getPointerPosition(event);

        lastX = position.x;
        lastY = position.y;

        placeholder.classList.add("hidden");

        canvas.setPointerCapture(event.pointerId);
    }


    /* =====================================================
       DRAW
    ===================================================== */

    function draw(event) {

        if (!isDrawing) {
            return;
        }

        event.preventDefault();

        const position = getPointerPosition(event);

        ctx.beginPath();

        ctx.moveTo(lastX, lastY);

        ctx.lineTo(position.x, position.y);

        ctx.stroke();

        lastX = position.x;
        lastY = position.y;

        hasDrawing = true;
    }


    /* =====================================================
       STOP DRAWING
    ===================================================== */

    function stopDrawing(event) {

        if (!isDrawing) {
            return;
        }

        isDrawing = false;

        if (event.pointerId !== undefined) {

            try {
                canvas.releasePointerCapture(event.pointerId);
            } catch (error) {
                // Pointer capture may already be released.
            }
        }
    }


    /* =====================================================
       CANVAS EVENTS
    ===================================================== */

    canvas.addEventListener(
        "pointerdown",
        startDrawing
    );

    canvas.addEventListener(
        "pointermove",
        draw
    );

    canvas.addEventListener(
        "pointerup",
        stopDrawing
    );

    canvas.addEventListener(
        "pointercancel",
        stopDrawing
    );

    canvas.addEventListener(
        "pointerleave",
        stopDrawing
    );


    /* =====================================================
       CLEAR BUTTON
    ===================================================== */

    clearBtn.addEventListener("click", () => {
        clearCanvas();
    });


    /* =====================================================
       ANALYZE BUTTON
    ===================================================== */

    analyzeBtn.addEventListener("click", async () => {

        hideError();

        if (!hasDrawing) {

            showError(
                "اكتب الحرف أولاً ثم اضغط على تحليل الكتابة."
            );

            return;
        }

        setLoading(true);

        try {

            const imageData = canvas.toDataURL("image/png");

            const response = await fetch(
                "/api/handwriting/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        image: imageData,
                        character: getTargetCharacter()
                    })
                }
            );


            const data = await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error || "حدث خطأ أثناء تحليل الكتابة."
                );
            }


            displayResult(data);

        } catch (error) {

            console.error(
                "Handwriting prediction error:",
                error
            );

            showError(
                error.message ||
                "تعذر تحليل الكتابة، حاول مرة أخرى."
            );

        } finally {

            setLoading(false);
        }
    });


    /* =====================================================
       RETRY
    ===================================================== */

    retryBtn.addEventListener("click", () => {

        clearCanvas();

        resultSection.classList.add("hidden");

        canvas.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    });


    /* =====================================================
       GET TARGET CHARACTER
    ===================================================== */

    function getTargetCharacter() {

        const target = document.getElementById(
            "targetCharacter"
        );

        return target.textContent.trim();
    }


    /* =====================================================
       DISPLAY RESULT
    ===================================================== */

    function displayResult(data) {

        /*
         Expected backend response:

         {
             "score": 92,
             "message": "...",
             "feedback": "...",
             "correct": true
         }
        */

        const score = Number(data.score || 0);

        scoreValue.textContent =
            `${Math.round(score)}%`;


        if (data.correct === true || score >= 80) {

            resultTitle.textContent = "أحسنت!";

            resultIcon.textContent = "✓";

            resultMessage.textContent =
                data.message ||
                "كتابتك ممتازة!";

            feedbackText.textContent =
                data.feedback ||
                "استمر بالتدريب وحافظ على نفس الطريقة.";

        } else if (score >= 50) {

            resultTitle.textContent = "محاولة جيدة";

            resultIcon.textContent = "↻";

            resultMessage.textContent =
                data.message ||
                "أنت قريب! حاول تحسين شكل الحرف.";

            feedbackText.textContent =
                data.feedback ||
                "ركز على شكل الحرف وحاول كتابته ببطء.";

        } else {

            resultTitle.textContent = "حاول مرة أخرى";

            resultIcon.textContent = "↻";

            resultMessage.textContent =
                data.message ||
                "لا بأس، جرب كتابة الحرف مرة أخرى.";

            feedbackText.textContent =
                data.feedback ||
                "خذ وقتك وحاول تقليد شكل الحرف الظاهر أمامك.";
        }


        resultSection.classList.remove("hidden");


        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }


    /* =====================================================
       LOADING
    ===================================================== */

    function setLoading(loading) {

        analyzeBtn.disabled = loading;

        clearBtn.disabled = loading;

        if (loading) {

            analyzeText.classList.add("hidden");

            analyzeLoader.classList.remove("hidden");

        } else {

            analyzeText.classList.remove("hidden");

            analyzeLoader.classList.add("hidden");
        }
    }


    /* =====================================================
       ERROR
    ===================================================== */

    function showError(message) {

        errorBox.textContent = message;

        errorBox.classList.remove("hidden");
    }


    function hideError() {

        errorBox.textContent = "";

        errorBox.classList.add("hidden");
    }


    /* =====================================================
       RESIZE
    ===================================================== */

    window.addEventListener("resize", () => {

        /*
         * Recreating the canvas on resize would erase
         * the current drawing, so only resize when empty.
         */

        if (!hasDrawing) {
            setupCanvas();
        }
    });


    /* =====================================================
       INITIALIZE
    ===================================================== */

    setupCanvas();

});