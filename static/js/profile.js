document.addEventListener("DOMContentLoaded", async () => {

    try {

        const response = await fetch("/api/profile");

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.message || "تعذر تحميل بيانات الملف الشخصي."
            );
        }

        const user = data.user;

        // =====================================================
        // Basic Information
        // =====================================================

        const name = user.name || "أ";

        document.getElementById("profileName").textContent = name;

        document.getElementById("profileAvatar").textContent =
            name.charAt(0);


        // =====================================================
        // Progress
        // =====================================================

        const progress = user.progress || {};

        const stories =
            Number(progress.stories || 0);

        const spelling =
            Number(progress.spelling || 0);

        const wordGame =
            Number(progress.word_game || 0);

        const pronunciation =
            Number(progress.pronunciation || 0);

        const handwriting =
            Number(progress.handwriting || 0);


        // =====================================================
        // Total Activities
        // =====================================================

        const totalActivities =
            stories +
            spelling +
            wordGame +
            pronunciation +
            handwriting;


        // =====================================================
        // Score
        // =====================================================

        // كل نشاط = 10 نقاط
        const totalScore =
            totalActivities * 10;

        document.getElementById("totalScore").textContent =
            totalScore;


        // =====================================================
        // Level
        // =====================================================

        let level = "مبتدئ";

        if (totalScore >= 100) {
            level = "متقدم";
        } else if (totalScore >= 50) {
            level = "متوسط";
        }

        document.getElementById("userLevel").textContent =
            level;


        // =====================================================
        // Skills
        // =====================================================

        // نحول عدد الأنشطة إلى نسبة.
        // كل 10 أنشطة = 100%
        const getPercentage = (count) => {

            return Math.min(
                Math.round((count / 10) * 100),
                100
            );

        };


        const readingPercentage =
            getPercentage(stories);

        const spellingPercentage =
            getPercentage(spelling);

        const pronunciationPercentage =
            getPercentage(pronunciation);

        const handwritingPercentage =
            getPercentage(handwriting);


        // =====================================================
        // Reading
        // =====================================================

        document.getElementById("readingProgress").textContent =
            `${readingPercentage}%`;

        document.getElementById("readingBar").style.width =
            `${readingPercentage}%`;


        // =====================================================
        // Spelling
        // =====================================================

        document.getElementById("spellingProgress").textContent =
            `${spellingPercentage}%`;

        document.getElementById("spellingBar").style.width =
            `${spellingPercentage}%`;


        // =====================================================
        // Pronunciation
        // =====================================================

        document.getElementById("pronunciationProgress").textContent =
            `${pronunciationPercentage}%`;

        document.getElementById("pronunciationBar").style.width =
            `${pronunciationPercentage}%`;


        // =====================================================
        // Handwriting
        // =====================================================

        document.getElementById("handwritingProgress").textContent =
            `${handwritingPercentage}%`;

        document.getElementById("handwritingBar").style.width =
            `${handwritingPercentage}%`;


        // =====================================================
        // Statistics
        // =====================================================

        document.getElementById("storiesCount").textContent =
            stories;

        document.getElementById("activitiesCount").textContent =
            totalActivities;

        // مؤقتًا
        document.getElementById("daysCount").textContent =
            totalActivities > 0 ? 1 : 0;

        document.getElementById("bestScore").textContent =
            totalActivities > 0 ? 100 : 0;


    } catch (error) {

        console.error(
            "[Profile] Error:",
            error
        );

    }

});