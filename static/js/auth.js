document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");

    // =========================================================
    // Helpers
    // =========================================================
    function showMessage(message, type = "error") {
        const messageBox = document.getElementById("formMessage");
        if (!messageBox) {return;}

        messageBox.textContent = message || "";
        messageBox.classList.remove("message-success", "message-error");

        if (message) {
            messageBox.classList.add(
                type === "success"? "message-success" : "message-error"
            );
        }
    }

    function setLoading(button, loading, defaultText) {
        if (!button) {return;}

        button.disabled = loading;
        if (loading) {
            button.dataset.originalText = button.textContent;
            button.textContent = "جاري التحميل...";
        } else {
            button.textContent = button.dataset.originalText || defaultText;
        }
    }

    async function sendRequest(url, data) {
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        let result;
        try {
            result = await response.json();
        } catch (error) {
            throw new Error("حدث خطأ غير متوقع من الخادم.");
        }
        return {response, result};
    }

    // =========================================================
    // LOGIN
    // =========================================================

    if (loginForm) {
        loginForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            showMessage("");
            const button = document.getElementById("loginButton");
            const username = document.getElementById("username").value.trim();
            const password = document.getElementById("password").value;

            if (!username || !password) {
                showMessage("يرجى إدخال اسم المستخدم وكلمة المرور.");
                return;
            }

            setLoading(button, true, "تسجيل الدخول");

            try {
                const { response, result } = await sendRequest("/login", {username, password});

                if (!response.ok || !result.success) {
                    showMessage(result.message || "اسم المستخدم أو كلمة المرور غير صحيحة.");
                    return;
                }

                showMessage("تم تسجيل الدخول بنجاح.", "success");

                // Go to home page
                window.location.href = "/";

            } catch (error) {
                console.error(error);
                showMessage("تعذر الاتصال بالخادم. حاول مرة أخرى.");

            } finally {
                setLoading(button, false, "تسجيل الدخول");
            }
        });
    }

    // =========================================================
    // REGISTER
    // =========================================================

    if (registerForm) {
        registerForm.addEventListener(
            "submit",
            async (event) => {

                event.preventDefault();
                showMessage("");
                const button = document.getElementById("registerButton");
                const name = document.getElementById("name").value.trim();
                const username = document.getElementById("username").value.trim();
                const password = document.getElementById("password").value;
                const confirmPassword = document.getElementById("confirmPassword").value;

                // -------------------------------------------------
                // Validation
                // -------------------------------------------------
                if (!username || !password) {
                    showMessage("اسم المستخدم وكلمة المرور مطلوبة.");
                    return;
                }

                if (password.length < 4) {
                    showMessage("كلمة المرور قصيرة جدًا.");
                    return;
                }

                if (password !== confirmPassword) {
                    showMessage("كلمتا المرور غير متطابقتين.");
                    return;
                }

                setLoading(button, true, "إنشاء الحساب");

                try {
                    const { response, result } = await sendRequest("/register", {name, username, password});

                    if (!response.ok || !result.success) {
                        showMessage(result.message || "تعذر إنشاء الحساب.");
                        return;
                    }

                    showMessage("تم إنشاء الحساب بنجاح.", "success");
                    window.location.href = "/";

                } catch (error) {
                    console.error(error);
                    showMessage("تعذر الاتصال بالخادم. حاول مرة أخرى.");

                } finally {
                    setLoading(button, false, "إنشاء الحساب");
                }
            }
        );
    }
});