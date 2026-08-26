// ============================================================
// PRONUNCIATION GAME
// ============================================================

let letters = [];
let selectedLetter = null;

let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let recordedAudioURL = null;

let recordingStartTime = null;
let recordingTimerInterval = null;


// ============================================================
// DOM ELEMENTS
// ============================================================

const letterSelection =
    document.getElementById("letterSelection");

const lettersLoading =
    document.getElementById("lettersLoading");

const lettersContainer =
    document.getElementById("lettersContainer");

const lettersError =
    document.getElementById("lettersError");

const lettersErrorText =
    document.getElementById("lettersErrorText");

const lettersRetryButton =
    document.getElementById("lettersRetryButton");


const practiceSection =
    document.getElementById("practiceSection");

const selectedLetterElement =
    document.getElementById("selectedLetter");

const referenceAudio =
    document.getElementById("referenceAudio");

const playReferenceButton =
    document.getElementById("playReferenceButton");

const audioButtonText =
    document.getElementById("audioButtonText");


const recordingStatus =
    document.getElementById("recordingStatus");

const recordingTimer =
    document.getElementById("recordingTimer");

const recordButton =
    document.getElementById("recordButton");

const recordButtonText =
    document.getElementById("recordButtonText");


const recordedAudioBox =
    document.getElementById("recordedAudioBox");

const recordedAudio =
    document.getElementById("recordedAudio");

const evaluateButton =
    document.getElementById("evaluateButton");


const recordingError =
    document.getElementById("recordingError");

const recordingErrorText =
    document.getElementById("recordingErrorText");


const evaluationResult =
    document.getElementById("evaluationResult");

const resultIcon =
    document.getElementById("resultIcon");

const resultLabel =
    document.getElementById("resultLabel");

const resultTitle =
    document.getElementById("resultTitle");

const resultMessage =
    document.getElementById("resultMessage");

const scoreValue =
    document.getElementById("scoreValue");


const tryAgainButton =
    document.getElementById("tryAgainButton");

const chooseAnotherButton =
    document.getElementById("chooseAnotherButton");


const evaluationLoading =
    document.getElementById("evaluationLoading");


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    loadLetters();
});


// ============================================================
// LOAD LETTERS
// ============================================================

async function loadLetters() {

    showElement(lettersLoading);
    hideElement(lettersError);

    lettersContainer.innerHTML = "";

    try {

        const response = await fetch(
            "/api/pronunciation/letters",
            {
                method: "GET",
                credentials: "same-origin"
            }
        );

        if (!response.ok) {
            throw new Error("تعذر تحميل الحروف.");
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(
                data.message || "تعذر تحميل الحروف."
            );
        }

        letters = data.letters || [];

        hideElement(lettersLoading);

        if (letters.length === 0) {

            showLettersError(
                "لا توجد حروف متاحة حاليًا."
            );

            return;
        }

        displayLetters();

    } catch (error) {

        console.error(
            "Load pronunciation letters error:",
            error
        );

        hideElement(lettersLoading);

        showLettersError(
            error.message || "تعذر تحميل الحروف."
        );
    }
}


// ============================================================
// DISPLAY LETTERS
// ============================================================

function displayLetters() {

    lettersContainer.innerHTML = "";

    letters.forEach((item, index) => {

        const button =
            document.createElement("button");

        button.type = "button";
        button.className = "letter-button";

        button.dataset.letter = item.letter;

        const letter =
            document.createElement("span");

        letter.className = "letter-character";
        letter.textContent = item.letter;

        const label =
            document.createElement("span");

        label.className = "letter-label";
        label.textContent = "تدرب";

        button.appendChild(letter);
        button.appendChild(label);

        button.addEventListener(
            "click",
            () => selectLetter(item)
        );

        lettersContainer.appendChild(button);
    });
}


// ============================================================
// SELECT LETTER
// ============================================================

function selectLetter(item) {

    selectedLetter = item;

    selectedLetterElement.textContent =
        item.letter;

    // Reference audio
    referenceAudio.src = item.audio;
    referenceAudio.load();

    // Reset everything
    resetRecording();

    hideElement(evaluationResult);
    hideElement(recordingError);

    recordingStatus.textContent =
        "اضغط على الزر وابدأ بنطق الحرف.";

    recordButton.classList.remove(
        "recording"
    );

    recordButtonText.textContent =
        "ابدأ التسجيل";

    audioButtonText.textContent =
        "استمع للنطق";

    // Show practice section
    hideElement(letterSelection);
    showElement(practiceSection);

    // Scroll to practice
    practiceSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// ============================================================
// PLAY REFERENCE AUDIO
// ============================================================

playReferenceButton.addEventListener(
    "click",
    async () => {

        if (!referenceAudio.src) {
            return;
        }

        try {

            if (!referenceAudio.paused) {

                referenceAudio.pause();

                audioButtonText.textContent =
                    "استمع للنطق";

                playReferenceButton.classList.remove(
                    "playing"
                );

                return;
            }

            await referenceAudio.play();

            audioButtonText.textContent =
                "جاري التشغيل...";

            playReferenceButton.classList.add(
                "playing"
            );

        } catch (error) {

            console.error(
                "Reference audio error:",
                error
            );

            showRecordingError(
                "تعذر تشغيل النطق الصحيح."
            );
        }
    }
);


// ============================================================
// AUDIO ENDED
// ============================================================

referenceAudio.addEventListener(
    "ended",
    () => {

        audioButtonText.textContent =
            "استمع للنطق";

        playReferenceButton.classList.remove(
            "playing"
        );
    }
);


// ============================================================
// RECORD BUTTON
// ============================================================

recordButton.addEventListener(
    "click",
    async () => {

        if (
            mediaRecorder &&
            mediaRecorder.state === "recording"
        ) {

            stopRecording();

        } else {

            await startRecording();
        }
    }
);


// ============================================================
// START RECORDING
// ============================================================

async function startRecording() {

    hideElement(recordingError);
    hideElement(evaluationResult);

    if (!navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia) {

        showRecordingError(
            "المتصفح لا يدعم استخدام المايك."
        );

        return;
    }

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                audio: true
            });

        audioChunks = [];
        recordedBlob = null;

        mediaRecorder =
            createMediaRecorder(stream);

        mediaRecorder.start();

        recordButton.classList.add(
            "recording"
        );

        recordButtonText.textContent =
            "إيقاف التسجيل";

        recordingStatus.textContent =
            "جاري التسجيل... انطق الحرف بوضوح.";

        showElement(recordingTimer);

        startRecordingTimer();

        hideElement(recordedAudioBox);
        hideElement(evaluateButton);

    } catch (error) {

        console.error(
            "Microphone error:",
            error
        );

        if (error.name === "NotAllowedError") {

            showRecordingError(
                "يجب السماح باستخدام المايك حتى تتمكن من تسجيل نطقك."
            );

        } else {

            showRecordingError(
                "تعذر فتح المايك. حاول مرة أخرى."
            );
        }
    }
}


// ============================================================
// CREATE MEDIA RECORDER
// ============================================================

function createMediaRecorder(stream) {

    let options = {};

    if (
        MediaRecorder.isTypeSupported(
            "audio/webm;codecs=opus"
        )
    ) {

        options.mimeType =
            "audio/webm;codecs=opus";

    } else if (
        MediaRecorder.isTypeSupported(
            "audio/webm"
        )
    ) {

        options.mimeType =
            "audio/webm";

    } else if (
        MediaRecorder.isTypeSupported(
            "audio/ogg;codecs=opus"
        )
    ) {

        options.mimeType =
            "audio/ogg;codecs=opus";
    }

    const recorder =
        new MediaRecorder(
            stream,
            options
        );

    recorder.addEventListener(
        "dataavailable",
        event => {

            console.log(
                "[Recorder] dataavailable:",
                event.data.size,
                event.data.type
            );

            if (
                event.data &&
                event.data.size > 0
            ) {
                audioChunks.push(event.data);
            }
        }
    );

    recorder.addEventListener(
        "stop",
        () => {

            console.log(
                "[Recorder] stopped. Chunks:",
                audioChunks.length
            );

            stream.getTracks().forEach(
                track => track.stop()
            );

            finishRecording();
        }
    );

    return recorder;
}


// ============================================================
// STOP RECORDING
// ============================================================

function stopRecording() {

    if (
        !mediaRecorder ||
        mediaRecorder.state !== "recording"
    ) {
        return;
    }

    mediaRecorder.requestData();
    mediaRecorder.stop();
    stopRecordingTimer();

    recordButton.classList.remove(
        "recording"
    );

    recordButtonText.textContent =
        "ابدأ التسجيل";

    recordingStatus.textContent =
        "تم التسجيل! يمكنك الاستماع إليه أو تقييمه.";

    hideElement(recordingTimer);
}


// ============================================================
// FINISH RECORDING
// ============================================================

function finishRecording() {

    console.log(
        "[Recorder] finishRecording"
    );

    console.log(
        "[Recorder] chunks:",
        audioChunks.length
    );

    if (audioChunks.length === 0) {

        showRecordingError(
            "لم يتم تسجيل أي صوت. حاول مرة أخرى."
        );

        return;
    }

    const mimeType =
        mediaRecorder?.mimeType ||
        audioChunks[0]?.type ||
        "audio/webm";

    console.log(
        "[Recorder] mimeType:",
        mimeType
    );

    recordedBlob =
        new Blob(
            audioChunks,
            {
                type: mimeType
            }
        );

    console.log(
        "[Recorder] blob size:",
        recordedBlob.size
    );

    console.log(
        "[Recorder] blob type:",
        recordedBlob.type
    );

    if (recordedBlob.size === 0) {

        showRecordingError(
            "التسجيل فارغ. حاول مرة أخرى."
        );

        return;
    }

    if (recordedAudioURL) {

        URL.revokeObjectURL(
            recordedAudioURL
        );
    }

    recordedAudioURL =
        URL.createObjectURL(
            recordedBlob
        );

    recordedAudio.src =
        recordedAudioURL;

    recordedAudio.load();

    showElement(recordedAudioBox);
    showElement(evaluateButton);

    // Check browser audio metadata
    recordedAudio.onloadedmetadata = () => {

        console.log(
            "[Recorder] duration:",
            recordedAudio.duration
        );

        if (
            !isFinite(recordedAudio.duration) ||
            recordedAudio.duration <= 0
        ) {

            console.warn(
                "[Recorder] Audio duration could not be detected."
            );
        }
    };
}


// ============================================================
// RECORDING TIMER
// ============================================================

function startRecordingTimer() {

    recordingStartTime =
        Date.now();

    recordingTimerInterval =
        setInterval(() => {

            const elapsed =
                Math.floor(
                    (Date.now() -
                        recordingStartTime) / 1000
                );

            const minutes =
                Math.floor(elapsed / 60);

            const seconds =
                elapsed % 60;

            recordingTimer.textContent =
                `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

        }, 250);
}


function stopRecordingTimer() {

    if (recordingTimerInterval) {

        clearInterval(
            recordingTimerInterval
        );

        recordingTimerInterval = null;
    }
}


// ============================================================
// EVALUATE PRONUNCIATION
// ============================================================

evaluateButton.addEventListener(
    "click",
    async () => {

        if (!recordedBlob) {

            showRecordingError(
                "سجّل صوتك أولًا."
            );

            return;
        }

        if (!selectedLetter) {

            showRecordingError(
                "اختر حرفًا أولًا."
            );

            return;
        }

        await evaluatePronunciation();
    }
);


// ============================================================
// SEND AUDIO TO SERVER
// ============================================================

async function evaluatePronunciation() {

    hideElement(recordingError);

    showElement(evaluationLoading);

    evaluateButton.disabled = true;
    recordButton.disabled = true;

    try {

        const formData =
            new FormData();

        formData.append(
            "target_letter",
            selectedLetter.letter
        );

        const extension =
            getAudioExtension(
                recordedBlob.type
            );

        const audioFile =
            new File(
                [recordedBlob],
                `pronunciation.${extension}`,
                {
                    type:
                        recordedBlob.type ||
                        "audio/webm"
                }
            );

        formData.append(
            "audio",
            audioFile
        );

        const response =
            await fetch(
                "/api/pronunciation/evaluate",
                {
                    method: "POST",
                    credentials: "same-origin",
                    body: formData
                }
            );

        if (!response.ok) {

            let message =
                "تعذر تقييم النطق.";

            try {

                const errorData =
                    await response.json();

                message =
                    errorData.message ||
                    message;

            } catch (_) {}

            throw new Error(message);
        }

        const data =
            await response.json();

        if (!data.success) {

            throw new Error(
                data.message ||
                "تعذر تقييم النطق."
            );
        }

        displayResult(
            data.result
        );

    } catch (error) {

        console.error(
            "Pronunciation evaluation error:",
            error
        );

        showRecordingError(
            error.message ||
            "حدث خطأ أثناء تقييم النطق."
        );

    } finally {

        hideElement(evaluationLoading);

        evaluateButton.disabled = false;
        recordButton.disabled = false;
    }
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(result) {

    const score =
        Number(result.score || 0);

    const correct =
        Boolean(result.correct);

    scoreValue.textContent =
        Math.round(score);

    resultIcon.textContent =
        correct ? "✓" : "!";

    resultLabel.textContent =
        correct
            ? "نطق صحيح"
            : "نحتاج إلى محاولة أخرى";

    resultTitle.textContent =
        correct
            ? getPositiveTitle(score)
            : getTryAgainTitle(score);

    resultMessage.textContent =
        correct
            ? getPositiveMessage(score)
            : getTryAgainMessage(score);

    evaluationResult.classList.remove(
        "result-correct",
        "result-wrong"
    );

    evaluationResult.classList.add(
        correct
            ? "result-correct"
            : "result-wrong"
    );

    showElement(evaluationResult);

    evaluationResult.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


// ============================================================
// POSITIVE MESSAGES
// ============================================================

function getPositiveTitle(score) {

    if (score >= 95) {
        return "ممتاز جدًا! 🌟";
    }

    if (score >= 85) {
        return "أحسنت! 👏";
    }

    return "رائع! 💪";
}


function getPositiveMessage(score) {

    if (score >= 95) {

        return "نطقك ممتاز وواضح جدًا! استمر بهذا المستوى.";

    }

    if (score >= 85) {

        return "نطقك جميل جدًا! واصل التدريب لتصبح أفضل.";

    }

    return "نطقك جيد! مع المزيد من التدريب ستتحسن أكثر.";
}


// ============================================================
// TRY AGAIN MESSAGES
// ============================================================

function getTryAgainTitle(score) {

    if (score >= 60) {
        return "قريب جدًا! 💪";
    }

    if (score >= 40) {
        return "محاولة جميلة! 🌱";
    }

    return "لا بأس! حاول مرة أخرى 🎯";
}


function getTryAgainMessage(score) {

    if (score >= 60) {

        return "أنت قريب من النطق الصحيح! استمع للنطق مرة أخرى وحاول تقليده.";

    }

    if (score >= 40) {

        return "أعد الاستماع إلى النطق الصحيح ثم حاول نطق الحرف بوضوح.";

    }

    return "لا تقلق! استمع إلى النطق الصحيح وحاول مرة أخرى.";
}


// ============================================================
// TRY AGAIN
// ============================================================

tryAgainButton.addEventListener(
    "click",
    () => {

        hideElement(evaluationResult);

        resetRecording();

        recordingStatus.textContent =
            "استمع للنطق الصحيح ثم حاول مرة أخرى.";

        practiceSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
);


// ============================================================
// CHOOSE ANOTHER LETTER
// ============================================================

chooseAnotherButton.addEventListener(
    "click",
    () => {

        resetRecording();

        selectedLetter = null;

        referenceAudio.pause();
        referenceAudio.removeAttribute(
            "src"
        );
        referenceAudio.load();

        hideElement(practiceSection);

        showElement(letterSelection);

        hideElement(evaluationResult);

        letterSelection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
);


// ============================================================
// RESET RECORDING
// ============================================================

function resetRecording() {

    if (
        mediaRecorder &&
        mediaRecorder.state === "recording"
    ) {

        mediaRecorder.stop();
    }
    mediaRecorder = null;

    stopRecordingTimer();

    audioChunks = [];
    recordedBlob = null;

    if (recordedAudioURL) {

        URL.revokeObjectURL(
            recordedAudioURL
        );

        recordedAudioURL = null;
    }

    recordedAudio.removeAttribute(
        "src"
    );

    recordedAudio.load();

    hideElement(recordedAudioBox);
    hideElement(evaluateButton);
    hideElement(recordingTimer);

    recordButton.classList.remove(
        "recording"
    );

    recordButtonText.textContent =
        "ابدأ التسجيل";

    evaluateButton.disabled = false;
    recordButton.disabled = false;
}


// ============================================================
// AUDIO EXTENSION
// ============================================================

function getAudioExtension(mimeType) {

    if (
        mimeType.includes("ogg")
    ) {
        return "ogg";
    }

    if (
        mimeType.includes("mp4")
    ) {
        return "m4a";
    }

    if (
        mimeType.includes("wav")
    ) {
        return "wav";
    }

    return "webm";
}


// ============================================================
// ERRORS
// ============================================================

function showLettersError(message) {

    lettersErrorText.textContent =
        message;

    showElement(lettersError);
}


function showRecordingError(message) {

    recordingErrorText.textContent =
        message;

    showElement(recordingError);
}


// ============================================================
// RETRY LETTERS
// ============================================================

lettersRetryButton.addEventListener(
    "click",
    () => {
        loadLetters();
    }
);


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