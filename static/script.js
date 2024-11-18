// Event listeners for upload and predict buttons
document.getElementById('uploadButton').addEventListener('click', handleUpload);
document.getElementById('predictButton').addEventListener('click', handlePredict);

// Function to handle audio file upload
async function handleUpload() {
    try {
        const audioFile = getAudioFile();
        if (!audioFile) {
            showStatusMessage("Please select an audio file.");
            return;
        }

        const formData = createFormData(audioFile);
        const result = await processAudio(formData);
        displayResults(result);

        showPredictButton();
        showStatusMessage("Audio uploaded successfully.");
    } catch (error) {
        showErrorMessage("Error uploading audio: " + error.message);
    }
}

// Function to handle fetching predictions
async function handlePredict() {
    try {
        const audioFile = getAudioFile();
        if (!audioFile) {
            showStatusMessage("No audio file uploaded for prediction.");
            return;
        }

        const formData = createFormData(audioFile);
        const result = await processAudio(formData);
        displayResults(result);

        showStatusMessage("Predictions fetched successfully.");
    } catch (error) {
        showErrorMessage("Prediction Error: " + error.message);
    }
}

// Helper function to get the selected audio file
function getAudioFile() {
    const audioFileInput = document.getElementById('audioInput');
    return audioFileInput.files[0];
}

// Helper function to create FormData object with audio file
function createFormData(audioFile) {
    const formData = new FormData();
    formData.append("audio", audioFile);
    return formData;
}

// Function to process audio file on the server
async function processAudio(formData) {
    const response = await fetch('/process_audio', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error("Network response was not ok");
    }

    return await response.json();
}

// Function to display results on the page
function displayResults(result) {
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = `
        <h2>Predictions:</h2>
        <p><strong>Emotion Probabilities:</strong> ${JSON.stringify(result["Emotion Probabilities"], null, 2)}</p>
        <p><strong>Transcription:</strong> ${result["Transcription"]}</p>
        <p><strong>LLM Interpretation:</strong> ${result["LLM Interpretation"]}</p>
    `;
}

// Helper functions for displaying messages
function showStatusMessage(message) {
    document.getElementById("status").innerText = message;
}

function showErrorMessage(message) {
    document.getElementById('response').innerText = message;
}

function showPredictButton() {
    document.getElementById("predictButton").style.display = 'block';
}
