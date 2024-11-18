document.getElementById('uploadButton').addEventListener('click', uploadAudio);
document.getElementById('predictButton').addEventListener('click', fetchPredictions); // Fetch predictions when clicked

// Function to handle audio file upload
async function uploadAudio() {
    const audioFileInput = document.getElementById('audioInput');
    const audioFile = audioFileInput.files[0];

    if (!audioFile) {
        document.getElementById("status").innerText = "Please select an audio file.";
        return;
    }

    const audioUrl = URL.createObjectURL(audioFile);
    document.getElementById('audioPlayback').src = audioUrl;

    const formData = new FormData();
    formData.append("audio", audioFile);

    try {
        const response = await fetch('/process_audio', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const result = await response.json();
        displayResults(result);
        
        // Show the Predict Emotion button after successful upload
        document.getElementById("predictButton").style.display = 'block'; 

        document.getElementById("status").innerText = "Audio uploaded successfully.";
    } catch (error) {
        document.getElementById('response').innerText = "Error: " + error.message;
    }
}

// Fetch predictions from the server
async function fetchPredictions() {
    const audioFileInput = document.getElementById('audioInput');
    const audioFile = audioFileInput.files[0];

    if (!audioFile) {
        document.getElementById("status").innerText = "No audio file uploaded for prediction.";
        return;
    }

    const formData = new FormData();
    formData.append("audio", audioFile);

    try {
        const response = await fetch('/process_audio', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const result = await response.json();
        displayResults(result);
        document.getElementById("status").innerText = "Predictions fetched successfully.";
    } catch (error) {
        document.getElementById('response').innerText = "Prediction Error: " + error.message;
    }
}

// Function to display results
function displayResults(result) {
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = `
        <h2>Predictions:</h2>
        <p><strong>Emotion Probabilities:</strong> ${JSON.stringify(result["Emotion Probabilities"], null, 2)}</p>
        <p><strong>Transcription:</strong> ${result["Transcription"]}</p>
        <p><strong>LLM Interpretation:</strong> ${result["LLM Interpretation"]}</p> <!-- Here -->
    `;
}
