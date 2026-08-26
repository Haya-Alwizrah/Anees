document.addEventListener("DOMContentLoaded", () => {

    // =========================================================
    // Elements
    // =========================================================
    const userName = document.getElementById("userName");
    const userAvatar = document.getElementById("userAvatar");
    const completedCount = document.getElementById("completedCount");
    const progressPercentage = document.getElementById("progressPercentage");
    const progressFill = document.getElementById("progressFill");
    const progressMessage = document.getElementById("progressMessage");

    // =========================================================
    // Progress Configuration
    // =========================================================
    const progressFeatures = [
        "stories",
        "spelling",
        "word_game",
        "pronunciation",
        "handwriting"
    ];

    // =========================================================
    // Load User
    // =========================================================

    async function loadUser() {
        try {
            const response = await fetch("/profile", {
                method: "GET",
                headers: {"Accept": "application/json"}
            });

            // Session expired
            if (response.status === 401) {
                window.location.href = "/login";
                return;
            }


            const data = await response.json();
            if (!response.ok || !data.success) {
                console.error(
                    data.message ||
                    "Unable to load user."
                );
                return;
            }

            updateUser(data.user);
            updateProgress(data.user.progress);

        } catch (error) {
            console.error("Error loading user:", error);
        }

    }

    // =========================================================
    // Update User UI
    // =========================================================

    function updateUser(user) {
        if (!user) {return;}

        const name = user.name || user.username || "صديقي";
        if (userName) {
            userName.textContent = name;
        }

        if (userAvatar) {
            const firstCharacter = name.trim().charAt(0);
            userAvatar.textContent = firstCharacter || "أ";
        }
    }

    // =========================================================
    // Calculate Progress
    // =========================================================

    function calculateProgress(progress) {
        if (!progress) {
            return {completed: 0, percentage: 0};
        }

        let completed = 0;
        progressFeatures.forEach(feature => {
            const value = Number(progress[feature] || 0);
            if (value > 0) {
                completed++;
            }
        });

        const percentage = Math.round((completed / progressFeatures.length) * 100);
        return {completed, percentage};
    }

    // =========================================================
    // Update Progress UI
    // =========================================================
    function updateProgress(progress) {
        const result = calculateProgress(progress);

        if (completedCount) {
            completedCount.textContent = result.completed;
        }

        if (progressPercentage) {
            progressPercentage.textContent = result.percentage;
        }

        if (progressFill) {
            progressFill.style.width = `${result.percentage}%`;
        }

        if (progressMessage) {
            if (result.percentage === 0) {
                progressMessage.textContent = "ابدأ أول نشاط لك اليوم!";

            } else if (result.percentage < 100) {
                progressMessage.textContent = "أحسنت! استمر في التعلم وأكمل باقي الأنشطة.";

            } else {
                progressMessage.textContent = "رائع! لقد أكملت جميع الأنشطة.";
            }
        }
    }

    // =========================================================
    // Daily Tips
    // =========================================================

    const tips = [
        "اقرأ بصوت عالٍ عندما تقرأ قصة، فهذا يساعدك على تحسين نطقك وفهمك للغة العربية.",
        "تعلم كلمة عربية جديدة كل يوم وحاول استخدامها في جملة.",
        "لا تخف من الخطأ، فالأخطاء تساعدك على التعلم بشكل أفضل.",
        "حاول أن تكتب الكلمات التي تتعلمها حتى تتذكرها بشكل أفضل.",
        "اقرأ الجملة كاملة قبل أن تختار إجابتك.",
        "التدرب قليلًا كل يوم أفضل من التعلم لفترة طويلة مرة واحدة."
    ];

    function showDailyTip() {
        const dailyTip = document.getElementById("dailyTip");
        if (!dailyTip) {return;}

        const today = new Date();
        const day = today.getDate();
        const tipIndex = day % tips.length;
        dailyTip.textContent = tips[tipIndex];
    }

    // =========================================================
    // Initialize
    // =========================================================
    showDailyTip();
    loadUser();
});