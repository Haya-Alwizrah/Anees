document.addEventListener("DOMContentLoaded", () => {

    // =========================================================
    // Elements
    // =========================================================

    const targetLetterElement =
        document.getElementById("targetLetter");

    const targetLetterNameElement =
        document.getElementById("targetLetterName");

    const nextLetterBtn =
        document.getElementById("nextLetterBtn");

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

    const resultSection =
        document.getElementById("resultSection");

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


    // =========================================================
    // State
    // =========================================================

    let selectedImage = null;
    let cameraStream = null;

    let targetLetter = "ب";


    // =========================================================
    // Arabic letters
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
    // Initialize
    // =========================================================

    setTargetLetter("ب");


    // =========================================================
    // Target Letter
    // =========================================================

    function setTargetLetter(letter) {

        const found = arabicLetters.find(
            item => item.letter === letter
        );

        if (!found) {
            console.error(
                "Unsupported target letter:",
                letter
            );

            return;
        }

        targetLetter = found.letter;

        targetLetterElement.textContent =
            found.letter;

        targetLetterNameElement.textContent =
            found.name;

        resetResult();
    }


    nextLetterBtn.addEventListener("click", () => {

        const currentIndex =
            arabicLetters.findIndex(
                item => item.letter === targetLetter
            );

        const nextIndex =
            (currentIndex + 1) % arabicLetters.length;

        setTargetLetter(
            arabicLetters[nextIndex].letter
        );

    });


    // =========================================================
    // Upload Image
    // =========================================================

    uploadBtn.addEventListener("click", () => {
        imageInput.click();
    });


    imageInput.addEventListener("change", (event) => {

        const file = event.target.files[0];

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

    });


    // =========================================================
    // Show Image Preview
    // =========================================================

    function showImagePreview(file) {

        const url =
            URL.createObjectURL(file);

        imagePreview.src = url;

        imagePreviewContainer.classList.remove(
            "hidden"
        );

        analyzeBtn.disabled = false;

        hideStatus();

        resultSection.classList.add("hidden");
    }


    // =========================================================
    // Remove Image
    // =========================================================

    removeImageBtn.addEventListener("click", () => {

        selectedImage = null;

        imageInput.value = "";

        imagePreview.src = "";

        imagePreviewContainer.classList.add(
            "hidden"
        );

        analyzeBtn.disabled = true;

        resetResult();

    });


    // =========================================================
    // Camera
    // =========================================================

    cameraBtn.addEventListener(
        "click",
        async () => {

            try {

                cameraStream =
                    await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: "environment"
                        },
                        audio: false
                    });

                cameraVideo.srcObject =
                    cameraStream;

                cameraContainer.classList.remove(
                    "hidden"
                );

                imagePreviewContainer.classList.add(
                    "hidden"
                );

            } catch (error) {

                console.error(
                    "Camera error:",
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
    // Capture Image
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
        }
    );


    function stopCamera() {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(track => track.stop());

            cameraStream = null;
        }

        cameraVideo.srcObject = null;

        cameraContainer.classList.add(
            "hidden"
        );
    }


    // =========================================================
    // Analyze Handwriting
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
                    "لم يتم تحديد الحرف المطلوب.",
                    "error"
                );

                return;
            }


            // -------------------------------------------------
            // FormData
            // -------------------------------------------------

            const formData =
                new FormData();

            formData.append(
                "image",
                selectedImage,
                selectedImage.name || "handwriting.png"
            );

            formData.append(
                "target_letter",
                targetLetter
            );

            formData.append(
                "adaptive",
                "false"
            );


            console.log(
                "[Handwriting] Sending:",
                {
                    image: selectedImage.name,
                    imageType: selectedImage.type,
                    imageSize: selectedImage.size,
                    targetLetter: targetLetter
                }
            );


            // -------------------------------------------------
            // UI
            // -------------------------------------------------

            analyzeBtn.disabled = true;

            analyzeBtn.textContent =
                "جاري تحليل الكتابة...";

            showStatus(
                "جاري تحليل الكتابة، يرجى الانتظار...",
                "loading"
            );

            resultSection.classList.add(
                "hidden"
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


                // -------------------------------------------------
                // Read response
                // -------------------------------------------------

                const data =
                    await response.json();


                console.log(
                    "[Handwriting] Server response:",
                    data
                );


                // -------------------------------------------------
                // Server Error
                // -------------------------------------------------

                if (!response.ok || !data.success) {

                    throw new Error(
                        data.message ||
                        `حدث خطأ في الخادم (${response.status})`
                    );
                }


                // -------------------------------------------------
                // Success
                // -------------------------------------------------

                displayResult(
                    data.result
                );

                showStatus(
                    "تم تحليل الكتابة بنجاح.",
                    "success"
                );


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
            Number(result.score || 0).toFixed(2);


        gradeValue.textContent =
            result.grade || "-";


        tierValue.textContent =
            result.tier || "-";


        resultTarget.textContent =
            result.target_letter || targetLetter;


        resultPredicted.textContent =
            result.predicted_letter || "-";


        resultCorrect.textContent =
            result.correct
                ? "صحيح"
                : "يحتاج إلى تحسين";


        // -------------------------------------------------
        // Top 3
        // -------------------------------------------------

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


                    const letter =
                        document.createElement("span");

                    letter.className =
                        "top3-letter";

                    letter.textContent =
                        item.letter;


                    const confidence =
                        document.createElement("span");

                    confidence.className =
                        "top3-confidence";

                    confidence.textContent =
                        `${Number(
                            item.confidence || 0
                        ).toFixed(2)}%`;


                    const rank =
                        document.createElement("span");

                    rank.className =
                        "top3-rank";

                    rank.textContent =
                        `#${index + 1}`;


                    row.appendChild(rank);
                    row.appendChild(letter);
                    row.appendChild(confidence);

                    top3List.appendChild(row);

                }
            );

        }


        resultSection.classList.remove(
            "hidden"
        );


        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }


    // =========================================================
    // Status
    // =========================================================

    function showStatus(message, type) {

        statusMessage.textContent =
            message;

        statusMessage.className =
            "status-message";

        if (type === "error") {
            statusMessage.classList.add(
                "error"
            );
        }

        if (type === "success") {
            statusMessage.classList.add(
                "success"
            );
        }

        if (type === "loading") {
            statusMessage.classList.add(
                "loading"
            );
        }

        statusMessage.classList.remove(
            "hidden"
        );
    }


    function hideStatus() {

        statusMessage.classList.add(
            "hidden"
        );

        statusMessage.textContent = "";

    }


    // =========================================================
    // Reset Result
    // =========================================================

    function resetResult() {

        resultSection.classList.add(
            "hidden"
        );

        scoreValue.textContent = "0";
        gradeValue.textContent = "-";
        tierValue.textContent = "-";
        resultTarget.textContent = "-";
        resultPredicted.textContent = "-";
        resultCorrect.textContent = "-";

        top3List.innerHTML = "";

        hideStatus();

    }

});