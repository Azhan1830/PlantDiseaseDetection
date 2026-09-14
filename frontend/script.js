const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const detectButton = document.getElementById("detectButton");

const loading = document.getElementById("loading");
const errorMessage = document.getElementById("errorMessage");
const resultCard = document.getElementById("resultCard");


// Select image
imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    // Check file type
    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError("Please select a JPG, JPEG or PNG image.");

        imageInput.value = "";

        return;
    }


    // Preview image
    const reader = new FileReader();

    reader.onload = function (event) {

        imagePreview.src = event.target.result;

        previewContainer.classList.remove("d-none");

        resultCard.classList.add("d-none");

        errorMessage.classList.add("d-none");
    };

    reader.readAsDataURL(file);
});


// Detect disease
detectButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {

        showError("Please select a plant leaf image first.");

        return;
    }


    // Reset UI
    errorMessage.classList.add("d-none");
    resultCard.classList.add("d-none");

    loading.classList.remove("d-none");

    detectButton.disabled = true;


    // Create form data
    const formData = new FormData();

    formData.append("file", file);


    try {

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Prediction failed."
            );
        }


        // Display result
        document.getElementById("plantName").textContent =
            data.plant;

        document.getElementById("diseaseName").textContent =
            data.disease;

        document.getElementById("status").textContent =
            data.status;

        document.getElementById("confidence").textContent =
            data.confidence + "%";

        document.getElementById("treatment").textContent =
            data.treatment;

        document.getElementById("prevention").textContent =
            data.prevention;


        // Confidence bar
        const confidenceBar =
            document.getElementById("confidenceBar");

        confidenceBar.style.width =
            data.confidence + "%";


        // Show result
        resultCard.classList.remove("d-none");

    }

    catch (error) {

        showError(
            "Unable to connect to the prediction server. " +
            error.message
        );

    }

    finally {

        loading.classList.add("d-none");

        detectButton.disabled = false;
    }

});


// Show error
function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove("d-none");
}