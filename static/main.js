console.log("main.js loaded — Active Validation System Initialized");

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    if (!form) return;

    form.addEventListener("submit", (e) => {
        let isValid = true;
        const errors = [];

        // 1. Validate Temperature
        const temp = document.getElementById("Temp");
        if (temp) {
            const val = parseFloat(temp.value);
            if (isNaN(val) || val < 15 || val > 50) {
                temp.style.borderColor = "var(--danger)";
                errors.push("Temperature must be between 15°C and 50°C.");
                isValid = false;
            } else {
                temp.style.borderColor = "var(--border)";
            }
        }

        // 2. Validate Humidity
        const humidity = document.getElementById("Humidity");
        if (humidity) {
            const val = parseFloat(humidity.value);
            if (isNaN(val) || val < 0 || val > 100) {
                humidity.style.borderColor = "var(--danger)";
                errors.push("Humidity must be between 0% and 100%.");
                isValid = false;
            } else {
                humidity.style.borderColor = "var(--border)";
            }
        }

        // 3. Validate Cloud Cover
        const cloudCover = document.getElementById("Cloud Cover");
        if (cloudCover) {
            const val = parseFloat(cloudCover.value);
            if (isNaN(val) || val < 0 || val > 100) {
                cloudCover.style.borderColor = "var(--danger)";
                errors.push("Cloud Cover must be between 0% and 100%.");
                isValid = false;
            } else {
                cloudCover.style.borderColor = "var(--border)";
            }
        }

        // 4. Validate Rainfall
        const annual = document.getElementById("ANNUAL");
        if (annual) {
            const val = parseFloat(annual.value);
            if (isNaN(val) || val < 0 || val > 10000) {
                annual.style.borderColor = "var(--danger)";
                errors.push("Annual Rainfall must be between 0 and 10,000 mm.");
                isValid = false;
            } else {
                annual.style.borderColor = "var(--border)";
            }
        }

        if (!isValid) {
            e.preventDefault();
            
            // Remove existing alert if any
            const existingAlert = document.querySelector(".alert-validation");
            if (existingAlert) existingAlert.remove();

            // Create new validation alert banner
            const alertDiv = document.createElement("div");
            alertDiv.className = "alert alert-error alert-validation";
            alertDiv.style.marginBottom = "24px";
            alertDiv.innerHTML = `<strong>Please correct the following errors:</strong><ul style="margin: 8px 0 0 16px; padding: 0;">${errors.map(err => `<li>${err}</li>`).join("")}</ul>`;
            
            form.parentNode.insertBefore(alertDiv, form);
            window.scrollTo({ top: alertDiv.offsetTop - 100, behavior: "smooth" });

            // Reset submit button state
            const submitBtn = document.getElementById("submit-btn");
            const btnText = document.getElementById("btn-text");
            const btnLoading = document.getElementById("btn-loading");
            if (submitBtn && btnText && btnLoading) {
                btnText.style.display = "inline";
                btnLoading.style.display = "none";
                submitBtn.disabled = false;
            }
        }
    });
});