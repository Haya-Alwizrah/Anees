/* =========================================================
   ANEES - STORY PAGE
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       ELEMENTS
    ====================================================== */

    const generateBtn =
        document.getElementById("generateStoryBtn");

    const generateBtnText =
        document.getElementById("generateBtnText");

    const generateLoader =
        document.getElementById("generateLoader");

    const storyLoading =
        document.getElementById("storyLoading");

    const storyResult =
        document.getElementById("storyResult");

    const questionsScreen =
        document.getElementById("questionsScreen");

    const storyError =
        document.getElementById("storyError");

    const storyContent =
        document.getElementById("storyContent");

    const storyTitle =
        document.getElementById("storyTitle");

    const difficultWordsSection =
        document.getElementById("difficultWordsSection");

    const difficultWordsContainer =
        document.getElementById("difficultWords");

    const questionsSection =
        document.getElementById("questionsSection");

    const questionsContainer =
        document.getElementById("questionsContainer");

    const startQuestionsBtn =
        document.getElementById("startQuestionsBtn");

    const newStoryFromQuestionsBtn =
        document.getElementById("newStoryFromQuestionsBtn");

    const topicSelect =
        document.getElementById("topic");

    const characterSelect =
        document.getElementById("character");

    const wordDefinition =
        document.getElementById("wordDefinition");

    const definitionWord =
        document.getElementById("definitionWord");

    const definitionText =
        document.getElementById("definitionText");

    const closeDefinition =
        document.getElementById("closeDefinition");

    /* =====================================================
       STORY GENERATOR CARD
    ====================================================== */

    const generatorCard =
        document.querySelector(".generator-card");


    /* =====================================================
       STATE
    ====================================================== */

    let currentWords = [];
    let currentQuestions = [];


    /* =====================================================
       HELPERS
    ====================================================== */

    function showError(message) {

        storyError.textContent = message;

        storyError.classList.remove("hidden");
    }


    function hideError() {

        storyError.classList.add("hidden");

        storyError.textContent = "";
    }


    function setLoading(isLoading) {

        generateBtn.disabled = isLoading;

        if (isLoading) {

            generateBtnText.textContent =
                "جاري إنشاء القصة...";

            generateLoader.classList.remove("hidden");

            storyLoading.classList.remove("hidden");

            storyResult.classList.add("hidden");

            questionsScreen.classList.add("hidden");

            startQuestionsBtn.classList.add("hidden");

            questionsSection.classList.add("hidden");

        } else {

            generateBtnText.textContent =
                "ابدأ القصة";

            generateLoader.classList.add("hidden");

            storyLoading.classList.add("hidden");
        }
    }


    /* =====================================================
       NORMALIZE WORDS
    ====================================================== */

    function normalizeWords(wordsData) {

        if (!wordsData) {
            return [];
        }

        let rawWords = wordsData;


        /* {"words": [...]} */

        if (
            typeof wordsData === "object" &&
            !Array.isArray(wordsData) &&
            wordsData.words !== undefined
        ) {

            rawWords = wordsData.words;
        }


        /* {"words": {"كلمة": "معنى"}} */

        if (
            typeof rawWords === "object" &&
            !Array.isArray(rawWords)
        ) {

            return Object.entries(rawWords)
                .map(([word, definition]) => ({
                    word: String(word).trim(),
                    definition: String(definition).trim()
                }))
                .filter(item => item.word);
        }


        /* [{"word": "...", "definition": "..."}] */

        if (Array.isArray(rawWords)) {

            return rawWords
                .map(item => {

                    if (typeof item === "string") {

                        return {
                            word: item.trim(),
                            definition: ""
                        };
                    }


                    if (
                        !item ||
                        typeof item !== "object"
                    ) {

                        return null;
                    }


                    const word =
                        item.word ||
                        item.term ||
                        item.keyword ||
                        "";


                    const definition =
                        item.definition ||
                        item.meaning ||
                        item.explanation ||
                        item.def ||
                        "";


                    return {
                        word: String(word).trim(),
                        definition: String(definition).trim()
                    };
                })
                .filter(
                    item =>
                        item &&
                        item.word
                );
        }


        return [];
    }


    /* =====================================================
       SHUFFLE
    ====================================================== */

    function shuffle(array) {

        const result = [...array];

        for (
            let i = result.length - 1;
            i > 0;
            i--
        ) {

            const j =
                Math.floor(
                    Math.random() * (i + 1)
                );


            [
                result[i],
                result[j]
            ] = [
                result[j],
                result[i]
            ];
        }

        return result;
    }


    /* =====================================================
       NORMALIZE STORY TEXT
    ====================================================== */

    function cleanStoryText(story) {

        if (!story) {
            return "";
        }

        return String(story)
            .replace(/\r\n/g, "\n")
            .replace(/\r/g, "\n")
            .trim();
    }


    /* =====================================================
       SHOW WORD DEFINITION
    ====================================================== */

    function showDefinition(
        word,
        definition
    ) {

        definitionWord.textContent =
            word;

        definitionText.textContent =
            definition ||
            "لا يوجد تعريف متاح لهذه الكلمة.";

        wordDefinition.classList.remove(
            "hidden"
        );
    }


    function hideDefinition() {

        wordDefinition.classList.add(
            "hidden"
        );

        document
            .querySelectorAll(
                ".difficult-word.active"
            )
            .forEach(element => {

                element.classList.remove(
                    "active"
                );
            });
    }


    /* =====================================================
       RENDER STORY WITH DIFFICULT WORDS
    ====================================================== */

    function renderStoryWithWords(
        story,
        words
    ) {

        storyContent.innerHTML = "";

        const text =
            cleanStoryText(story);


        if (!text) {
            return;
        }


        if (
            !words ||
            words.length === 0
        ) {

            storyContent.textContent =
                text;

            return;
        }


        /* Longest words first */

        const validWords =
            words
                .filter(
                    item => item.word
                )
                .sort(
                    (a, b) =>
                        b.word.length -
                        a.word.length
                );


        const escapedWords =
            validWords.map(item =>
                item.word.replace(
                    /[.*+?^${}()|[\]\\]/g,
                    "\\$&"
                )
            );


        if (
            escapedWords.length === 0
        ) {

            storyContent.textContent =
                text;

            return;
        }


        const pattern =
            `(${escapedWords.join("|")})`;


        const regex =
            new RegExp(
                pattern,
                "g"
            );


        let lastIndex = 0;
        let match;


        while (
            (match = regex.exec(text))
            !== null
        ) {

            /* Normal text before word */

            if (
                match.index >
                lastIndex
            ) {

                storyContent.appendChild(
                    document.createTextNode(
                        text.slice(
                            lastIndex,
                            match.index
                        )
                    )
                );
            }


            const matchedText =
                match[0];


            /* Find definition */

            const wordData =
                validWords.find(
                    item =>
                        item.word ===
                        matchedText
                );


            const fallback =
                wordData ||
                validWords.find(
                    item =>
                        item.word.trim() ===
                        matchedText.trim()
                );


            if (fallback) {

                const span =
                    document.createElement(
                        "span"
                    );


                span.className =
                    "difficult-word";


                span.textContent =
                    matchedText;


                span.dataset.word =
                    fallback.word;


                span.dataset.definition =
                    fallback.definition || "";


                span.addEventListener(
                    "click",
                    event => {

                        event.stopPropagation();


                        document
                            .querySelectorAll(
                                ".difficult-word.active"
                            )
                            .forEach(element => {

                                element.classList.remove(
                                    "active"
                                );
                            });


                        span.classList.add(
                            "active"
                        );


                        showDefinition(
                            fallback.word,
                            fallback.definition
                        );
                    }
                );


                storyContent.appendChild(
                    span
                );

            } else {

                storyContent.appendChild(
                    document.createTextNode(
                        matchedText
                    )
                );
            }


            lastIndex =
                match.index +
                matchedText.length;
        }


        /* Remaining text */

        if (
            lastIndex <
            text.length
        ) {

            storyContent.appendChild(
                document.createTextNode(
                    text.slice(
                        lastIndex
                    )
                )
            );
        }
    }


    /* =====================================================
       RENDER DIFFICULT WORD CHIPS
    ====================================================== */

    function renderDifficultWords(words) {

        difficultWordsContainer.innerHTML =
            "";


        if (
            !words ||
            words.length === 0
        ) {

            difficultWordsSection.classList.add(
                "hidden"
            );

            return;
        }


        difficultWordsSection.classList.remove(
            "hidden"
        );


        words.forEach(item => {

            const button =
                document.createElement(
                    "button"
                );


            button.type =
                "button";


            button.className =
                "word-chip";


            button.textContent =
                item.word;


            button.addEventListener(
                "click",
                () => {

                    showDefinition(
                        item.word,
                        item.definition
                    );


                    storyContent.scrollIntoView({
                        behavior: "smooth",
                        block: "center"
                    });
                }
            );


            difficultWordsContainer.appendChild(
                button
            );
        });
    }


    /* =====================================================
       RENDER QUESTIONS
    ====================================================== */

    function renderQuestions(questionData) {

        questionsContainer.innerHTML =
            "";

        questionsSection.classList.add(
            "hidden"
        );

        startQuestionsBtn.classList.add(
            "hidden"
        );


        if (!questionData) {
            return;
        }


        let questions =
            questionData.questions;


        /* Support direct array */

        if (
            Array.isArray(
                questionData
            )
        ) {

            questions =
                questionData;
        }


        if (
            !Array.isArray(
                questions
            ) ||
            questions.length === 0
        ) {

            return;
        }


        /* Keep only first 5 questions */

        currentQuestions =
            questions.slice(0, 5);


        /* Create question cards */

        currentQuestions.forEach(
            (question, index) => {

                createQuestionCard(
                    question,
                    index
                );
            }
        );


        /* Show "هل فهمت القصة؟" button */

        startQuestionsBtn.classList.remove(
            "hidden"
        );
    }


    /* =====================================================
       CREATE QUESTION CARD
    ====================================================== */

    function createQuestionCard(
        question,
        index
    ) {

        const card =
            document.createElement(
                "div"
            );


        card.className =
            "question-card";


        const number =
            document.createElement(
                "div"
            );


        number.className =
            "question-number";


        number.textContent =
            index + 1;


        const questionText =
            document.createElement(
                "div"
            );


        questionText.className =
            "question-text";


        questionText.textContent =
            question.question || "";


        const answers =
            document.createElement(
                "div"
            );


        answers.className =
            "answers";


        /* Correct answer */

        const correctAnswer =
            String(
                question.c_answer || ""
            ).trim();


        /* Wrong answers */

        const wrongAnswers =
            Array.isArray(
                question.w_answer
            )
                ? question.w_answer
                : [];


        /* Combine and shuffle */

        const options =
            shuffle(
                [
                    correctAnswer,
                    ...wrongAnswers
                ]
                .filter(Boolean)
            );


        const feedback =
            document.createElement(
                "div"
            );


        feedback.className =
            "question-feedback";


        options.forEach(answer => {

            const button =
                document.createElement(
                    "button"
                );


            button.type =
                "button";


            button.className =
                "answer-btn";


            button.textContent =
                answer;


            button.addEventListener(
                "click",
                () => {

                    handleAnswer(
                        button,
                        answer,
                        correctAnswer,
                        answers,
                        feedback
                    );
                }
            );


            answers.appendChild(
                button
            );
        });


        card.appendChild(
            number
        );


        card.appendChild(
            questionText
        );


        card.appendChild(
            answers
        );


        card.appendChild(
            feedback
        );


        questionsContainer.appendChild(
            card
        );
    }


    /* =====================================================
       HANDLE ANSWER
    ====================================================== */

    function handleAnswer(
        selectedButton,
        selectedAnswer,
        correctAnswer,
        answersContainer,
        feedback
    ) {

        /* Prevent changing answer */

        const buttons =
            answersContainer.querySelectorAll(
                ".answer-btn"
            );


        buttons.forEach(
            button => {

                button.disabled =
                    true;
            }
        );


        const isCorrect =
            selectedAnswer.trim() ===
            correctAnswer.trim();


        if (isCorrect) {

            selectedButton.classList.add(
                "correct"
            );


            feedback.textContent =
                "أحسنت! إجابة صحيحة 👏";


            feedback.className =
                "question-feedback correct-feedback";

        } else {

            selectedButton.classList.add(
                "wrong"
            );


            /* Show correct answer */

            buttons.forEach(
                button => {

                    if (
                        button.textContent.trim() ===
                        correctAnswer.trim()
                    ) {

                        button.classList.add(
                            "correct"
                        );
                    }
                }
            );


            feedback.textContent =
                `الإجابة الصحيحة هي: ${correctAnswer}`;


            feedback.className =
                "question-feedback wrong-feedback";
        }
    }


    /* =====================================================
       GENERATE STORY
    ====================================================== */

    async function generateStory() {

        hideError();


        const topic =
            topicSelect.value;


        const character =
            characterSelect.value;


        setLoading(true);


        try {

            const response =
                await fetch(
                    "/api/story/generate",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        credentials:
                            "same-origin",

                        body:
                            JSON.stringify({
                                topic:
                                    topic,

                                character:
                                    character
                            })
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.message ||
                    "حدث خطأ أثناء إنشاء القصة."
                );
            }


            /* Story */

            const story =
                data.story || "";


            /* Difficult words */

            currentWords =
                normalizeWords(
                    data.words
                );


            /* Questions */

            const questions =
                data.questions;


            /* Title */

            storyTitle.textContent =
                `قصة عن ${topic}`;


            /* Render story */

            renderStoryWithWords(
                story,
                currentWords
            );


            /* Render difficult words */

            renderDifficultWords(
                currentWords
            );


            /* Prepare questions */

            renderQuestions(
                questions
            );


            /* =================================================
               IMPORTANT:
               Hide generator after story is created.
            ================================================== */

            generatorCard.classList.add(
                "hidden"
            );


            /* Show story result */

            storyResult.classList.remove(
                "hidden"
            );


            /* Make sure questions screen is hidden */

            questionsScreen.classList.add(
                "hidden"
            );


            /* Questions section inside story is hidden */

            questionsSection.classList.add(
                "hidden"
            );


            /* Scroll to story */

            setTimeout(() => {

                storyResult.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            }, 100);


        } catch (error) {

            console.error(
                "Story generation error:",
                error
            );


            showError(
                error.message ||
                "تعذر إنشاء القصة. حاول مرة أخرى."
            );


            storyResult.classList.add(
                "hidden"
            );


            questionsScreen.classList.add(
                "hidden"
            );


            startQuestionsBtn.classList.add(
                "hidden"
            );


        } finally {

            setLoading(false);
        }
    }


    /* =====================================================
       START QUESTIONS
    ====================================================== */

    function startQuestions() {

        /*
         * Hide the story page completely.
         */
        storyResult.classList.add(
            "hidden"
        );


        /*
         * Hide "هل فهمت القصة؟" button.
         */
        startQuestionsBtn.classList.add(
            "hidden"
        );


        /*
         * Show questions screen.
         */
        questionsScreen.classList.remove(
            "hidden"
        );


        /*
         * Show questions section.
         */
        questionsSection.classList.remove(
            "hidden"
        );


        /*
         * Scroll to questions.
         */
        setTimeout(() => {

            questionsScreen.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }, 100);
    }


    /* =====================================================
       RESET PAGE / NEW STORY
    ====================================================== */

    function resetStory() {

        /*
         * Hide story result.
         */
        storyResult.classList.add(
            "hidden"
        );


        /*
         * Hide questions screen.
         */
        questionsScreen.classList.add(
            "hidden"
        );


        /*
         * Show generator again.
         */
        generatorCard.classList.remove(
            "hidden"
        );


        /*
         * Hide difficult words.
         */
        difficultWordsSection.classList.add(
            "hidden"
        );


        /*
         * Hide questions.
         */
        questionsSection.classList.add(
            "hidden"
        );


        /*
         * Hide questions button.
         */
        startQuestionsBtn.classList.add(
            "hidden"
        );


        /*
         * Clear content.
         */
        storyContent.innerHTML =
            "";

        difficultWordsContainer.innerHTML =
            "";

        questionsContainer.innerHTML =
            "";


        /*
         * Reset state.
         */
        currentWords = [];
        currentQuestions = [];


        /*
         * Hide definition and errors.
         */
        hideDefinition();
        hideError();


        /*
         * Scroll to top.
         */
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }


    /* =====================================================
       EVENTS
    ====================================================== */

    /*
     * Generate story.
     */
    generateBtn.addEventListener(
        "click",
        generateStory
    );


    /*
     * Start questions.
     */
    startQuestionsBtn.addEventListener(
        "click",
        startQuestions
    );


    /*
     * New story from questions.
     */
    newStoryFromQuestionsBtn.addEventListener(
        "click",
        resetStory
    );


    /*
     * Close definition.
     */
    closeDefinition.addEventListener(
        "click",
        event => {

            event.stopPropagation();

            hideDefinition();
        }
    );


    /*
     * Clicking outside definition
     * closes it.
     */
    document.addEventListener(
        "click",
        event => {

            if (
                !event.target.closest(
                    ".word-definition"
                )
                &&
                !event.target.closest(
                    ".difficult-word"
                )
                &&
                !event.target.closest(
                    ".word-chip"
                )
            ) {

                hideDefinition();
            }
        }
    );

});