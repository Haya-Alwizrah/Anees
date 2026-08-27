document.addEventListener("DOMContentLoaded", () => {

    // =========================================================
    // Elements
    // =========================================================

    const letterSelection =
        document.getElementById("letterSelection");

    const practiceSection =
        document.getElementById("practiceSection");

    const resultSection =
        document.getElementById("resultSection");

    const lettersContainer =
        document.getElementById("lettersContainer");

    const targetLetterElement =
        document.getElementById("targetLetter");

    const targetLetterNameElement =
        document.getElementById("targetLetterName");

    const backToLettersBtn =
        document.getElementById("backToLettersBtn");

    const cameraBtn =
        document.getElementById("cameraBtn");

    const uploadBtn =
        document.getElementById("uploadBtn");

    const imageInput =
        document.getElementById("imageInput");

    const imagePreviewContainer =
        document.getElementById("imagePreviewContainer");

    const imagePreview =
        document.getElementById("imagePreview");

    const removeImageBtn =
        document.getElementById("removeImageBtn");

    const uploadActions =
        document.getElementById("uploadActions");

    const cameraContainer =
        document.getElementById("cameraContainer");

    const cameraVideo =
        document.getElementById("cameraVideo");

    const cameraCanvas =
        document.getElementById("cameraCanvas");

    const captureBtn =
        document.getElementById("captureBtn");

    const closeCameraBtn =
        document.getElementById("closeCameraBtn");

    const analyzeBtn =
        document.getElementById("analyzeBtn");

    const statusMessage =
        document.getElementById("statusMessage");

    const scoreValue =
        document.getElementById("scoreValue");

    const gradeValue =
        document.getElementById("gradeValue");

    const tierValue =
        document.getElementById("tierValue");

    const resultTarget =
        document.getElementById("resultTarget");

    const resultPredicted =
        document.getElementById("resultPredicted");

    const resultCorrect =
        document.getElementById("resultCorrect");

    const top3List =
        document.getElementById("top3List");

    const retryBtn =
        document.getElementById("retryBtn");

    const chooseAnotherBtn =
        document.getElementById("chooseAnotherBtn");


    // =========================================================
    // State
    // =========================================================

    let selectedImage = null;
    let cameraStream = null;
    let targetLetter = null;


    // =========================================================
    // Arabic Letters
    // =========================================================

    const arabicLetters = [
        { letter: "ا", name: "حرف الألف" },
        { letter: "ب", name: "حرف الباء" },
        { letter: "ت", name: "حرف التاء" },
        { letter: "ث", name: "حرف الثاء" },
        { letter: "ج", name: "حرف الجيم" },
        { letter: "ح", name: "حرف الحاء" },
        { letter: "خ", name: "حرف الخاء" },
        { letter: "د", name: "حرف الدال" },
        { letter: "ذ", name: "حرف الذال" },
        { letter: "ر", name: "حرف الراء" },
        { letter: "ز", name: "حرف الزاي" },
        { letter: "س", name: "حرف السين" },
        { letter: "ش", name: "حرف الشين" },
        { letter: "ص", name: "حرف الصاد" },
        { letter: "ض", name: "حرف الضاد" },
        { letter: "ط", name: "حرف الطاء" },
        { letter: "ظ", name: "حرف الظاء" },
        { letter: "ع", name: "حرف العين" },
        { letter: "غ", name: "حرف الغين" },
        { letter: "ف", name: "حرف الفاء" },
        { letter: "ق", name: "حرف القاف" },
        { letter: "ك", name: "حرف الكاف" },
        { letter: "ل", name: "حرف اللام" },
        { letter: "م", name: "حرف الميم" },
        { letter: "ن", name: "حرف النون" },
        { letter: "ه", name: "حرف الهاء" },
        { letter: "و", name: "حرف الواو" },
        { letter: "ي", name: "حرف الياء" }
    ];


    // =========================================================
    // Create Letters
    // =========================================================

    arabicLetters.forEach((item) => {

        const button =
            document.createElement("button");

        button.type = "button";

        button.className =
            "letter-btn";

        button.innerHTML = `
            <span class="letter-character">
                ${item.letter}
            </span>

            <span class="letter-name">
                ${item.name.replace("حرف ", "")}
            </span>
        `;

        button.addEventListener("click", () => {
            selectLetter(item);
        });

        lettersContainer.appendChild(button);
    });


    // =========================================================
    // Select Letter
    // =========================================================

    function selectLetter(item) {

        targetLetter = item.letter;

        targetLetterElement.textContent =
            item.letter;

        targetLetterNameElement.textContent =
            item.name;

        resetImage();

        resetResult();

        letterSelection.classList.add("hidden");

        resultSection.classList.add("hidden");

        practiceSection.classList.remove("hidden");

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }


    // =========================================================
    // Back To Letters
    // =========================================================

    backToLettersBtn.addEventListener(
        "click",
        () => {

            stopCamera();

            resetImage();

            practiceSection.classList.add("hidden");

            resultSection.classList.add("hidden");

            letterSelection.classList.remove("hidden");

            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });
        }
    );


    // =========================================================
    // Upload
    // =========================================================

    uploadBtn.addEventListener(
        "click",
        () => {
            imageInput.click();
        }
    );


    imageInput.addEventListener(
        "change",
        (event) => {

            const file =
                event.target.files[0];

            if (!file) {
                return;
            }

            if (!file.type.startsWith("image/")) {

                showStatus(
                    "الملف المحدد ليس صورة.",
                    "error"
                );

                return;
            }

            selectedImage = file;

            showImagePreview(file);

            stopCamera();
        }
    );


    // =========================================================
    // Image Preview
    // =========================================================

    function showImagePreview(file) {

        const url =
            URL.createObjectURL(file);

        imagePreview.src = url;

        imagePreviewContainer
            .classList
            .remove("hidden");

        uploadActions
            .classList
            .add("hidden");

        analyzeBtn.disabled = false;

        hideStatus();

        resultSection.classList.add("hidden");
    }


    // =========================================================
    // Remove Image
    // =========================================================

    removeImageBtn.addEventListener(
        "click",
        () => {
            resetImage();
        }
    );


    function resetImage() {

        selectedImage = null;

        imageInput.value = "";

        imagePreview.src = "";

        imagePreviewContainer
            .classList
            .add("hidden");

        uploadActions
            .classList
            .remove("hidden");

        analyzeBtn.disabled = true;

        stopCamera();

        hideStatus();
    }


    // =========================================================
    // Camera
    // =========================================================

    cameraBtn.addEventListener(
        "click",
        async () => {

            try {

                if (!navigator.mediaDevices ||
                    !navigator.mediaDevices.getUserMedia) {

                    throw new Error(
                        "الكاميرا غير مدعومة في هذا المتصفح."
                    );
                }

                cameraStream =
                    await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: {
                                ideal: "environment"
                            }
                        },
                        audio: false
                    });

                cameraVideo.srcObject =
                    cameraStream;

                cameraContainer
                    .classList
                    .remove("hidden");

                uploadActions
                    .classList
                    .add("hidden");

                imagePreviewContainer
                    .classList
                    .add("hidden");

            } catch (error) {

                console.error(
                    "[Handwriting] Camera error:",
                    error
                );

                showStatus(
                    "تعذر تشغيل الكاميرا. تأكد من السماح باستخدام الكاميرا.",
                    "error"
                );
            }
        }
    );


    // =========================================================
    // Capture
    // =========================================================

    captureBtn.addEventListener(
        "click",
        () => {

            if (!cameraStream) {
                return;
            }

            const width =
                cameraVideo.videoWidth;

            const height =
                cameraVideo.videoHeight;

            if (!width || !height) {

                showStatus(
                    "انتظر حتى تظهر صورة الكاميرا ثم حاول مرة أخرى.",
                    "error"
                );

                return;
            }

            cameraCanvas.width = width;
            cameraCanvas.height = height;

            const context =
                cameraCanvas.getContext("2d");

            context.drawImage(
                cameraVideo,
                0,
                0,
                width,
                height
            );

            cameraCanvas.toBlob(
                (blob) => {

                    if (!blob) {

                        showStatus(
                            "تعذر التقاط الصورة.",
                            "error"
                        );

                        return;
                    }

                    selectedImage =
                        new File(
                            [blob],
                            "handwriting.png",
                            {
                                type: "image/png"
                            }
                        );

                    showImagePreview(
                        selectedImage
                    );

                    stopCamera();
                },
                "image/png"
            );
        }
    );


    // =========================================================
    // Close Camera
    // =========================================================

    closeCameraBtn.addEventListener(
        "click",
        () => {

            stopCamera();

            uploadActions
                .classList
                .remove("hidden");
        }
    );


    function stopCamera() {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

            cameraStream = null;
        }

        cameraVideo.srcObject = null;

        cameraContainer
            .classList
            .add("hidden");
    }


    // =========================================================
    // Analyze
    // =========================================================

    analyzeBtn.addEventListener(
        "click",
        async () => {

            if (!selectedImage) {

                showStatus(
                    "يرجى رفع صورة الكتابة أو التقاط صورة أولًا.",
                    "error"
                );

                return;
            }

            if (!targetLetter) {

                showStatus(
                    "يرجى اختيار الحرف أولًا.",
                    "error"
                );

                return;
            }


            const formData =
                new FormData();

            formData.append(
                "image",
                selectedImage,
                selectedImage.name ||
                "handwriting.png"
            );

            formData.append(
                "target_letter",
                targetLetter
            );

            formData.append(
                "adaptive",
                "false"
            );


            analyzeBtn.disabled = true;

            analyzeBtn.textContent =
                "جاري تحليل الكتابة...";

            showStatus(
                "جاري تحليل الكتابة، يرجى الانتظار...",
                "loading"
            );


            try {

                const response =
                    await fetch(
                        "/api/handwriting/evaluate",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok ||
                    !data.success) {

                    throw new Error(
                        data.message ||
                        `حدث خطأ في الخادم (${response.status})`
                    );
                }


                displayResult(
                    data.result
                );


                practiceSection
                    .classList
                    .add("hidden");

                resultSection
                    .classList
                    .remove("hidden");

                window.scrollTo({
                    top: 0,
                    behavior: "smooth"
                });


            } catch (error) {

                console.error(
                    "[Handwriting] Evaluation error:",
                    error
                );

                showStatus(
                    error.message ||
                    "حدث خطأ أثناء تحليل الكتابة.",
                    "error"
                );

            } finally {

                analyzeBtn.disabled =
                    !selectedImage;

                analyzeBtn.textContent =
                    "تحليل الكتابة";
            }
        }
    );


    // =========================================================
    // Display Result
    // =========================================================

    function displayResult(result) {

        if (!result) {
            return;
        }

        scoreValue.textContent =
            Number(
                result.score || 0
            ).toFixed(0);

        gradeValue.textContent =
            result.grade || "-";

        tierValue.textContent =
            result.tier || "-";

        resultTarget.textContent =
            result.target_letter ||
            targetLetter;

        resultPredicted.textContent =
            result.predicted_letter ||
            "-";


        resultCorrect.textContent =
            result.correct
                ? "صحيح ✓"
                : "يحتاج إلى تحسين";


        resultCorrect.className =
            result.correct
                ? "result-correct correct"
                : "result-correct incorrect";


        // Top 3
        top3List.innerHTML = "";


        if (
            Array.isArray(result.top3) &&
            result.top3.length > 0
        ) {

            result.top3.forEach(
                (item, index) => {

                    const row =
                        document.createElement("div");

                    row.className =
                        "top3-item";


                    row.innerHTML = `
                        <span class="top3-rank">
                            #${index + 1}
                        </span>

                        <span class="top3-letter">
                            ${item.letter}
                        </span>

                        <span class="top3-confidence">
                            ${Number(
                                item.confidence || 0
                            ).toFixed(2)}%
                        </span>
                    `;

                    top3List.appendChild(row);
                }
            );
        }
    }


    // =========================================================
    // Retry Same Letter
    // =========================================================

    retryBtn.addEventListener(
        "click",
        () => {

            resetImage();

            resetResult();

            resultSection
                .classList
                .add("hidden");

            practiceSection
                .classList
                .remove("hidden");

            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });
        }
    );


    // =========================================================
    // Choose Another Letter
    // =========================================================

    chooseAnotherBtn.addEventListener(
        "click",
        () => {

            resetImage();

            resetResult();

            resultSection
                .classList
                .add("hidden");

            practiceSection
                .classList
                .add("hidden");

            letterSelection
                .classList
                .remove("hidden");

            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });
        }
    );


    // =========================================================
    // Result Reset
    // =========================================================

    function resetResult() {

        scoreValue.textContent = "0";
        gradeValue.textContent = "-";
        tierValue.textContent = "-";

        resultTarget.textContent = "-";
        resultPredicted.textContent = "-";
        resultCorrect.textContent = "-";

        resultCorrect.className =
            "result-correct";

        top3List.innerHTML = "";

        resultSection
            .classList
            .add("hidden");

        hideStatus();
    }


    // =========================================================
    // Status
    // =========================================================

    function showStatus(message, type) {

        statusMessage.textContent =
            message;

        statusMessage.className =
            "status-message";

        if (type) {
            statusMessage.classList.add(type);
        }

        statusMessage
            .classList
            .remove("hidden");
    }


    function hideStatus() {

        statusMessage
            .classList
            .add("hidden");

        statusMessage.textContent = "";
    }

});