document.getElementById('audioUploadForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent the default form submission

    const audioFileInput = document.getElementById('audioFile');
    const audioFile = audioFileInput.files[0];

    // Clear previous messages
    document.getElementById('loadingMessage').style.display = 'none';
    document.getElementById('errorMessage').innerText = '';
    document.getElementById('results').style.display = 'none';
    document.getElementById('predictionResult').innerText = '';
    document.getElementById('predictEmotionButton').style.display = 'none';
    document.getElementById('makeMorePredictionsButton').style.display = 'none';

    if (!audioFile) {
        document.getElementById('errorMessage').innerText = "Please select an audio file.";
        return;
    }

    // Load audio for playback
    const audioUrl = URL.createObjectURL(audioFile);
    document.getElementById('audioPlayer').src = audioUrl;
    document.getElementById('audioPlayer').style.display = 'block'; // Show audio player
    document.getElementById('audioPlayer').play(); // Automatically play the audio

    const formData = new FormData();
    formData.append("audio", audioFile);

    // Show loading message
    document.getElementById('loadingMessage').style.display = 'block';
    document.getElementById('loadingMessage').innerText = "Processing your request...";

    try {
        // Send the audio file to the server
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
        document.getElementById("predictEmotionButton").style.display = 'block';
        document.getElementById("loadingMessage").style.display = 'none';

        document.getElementById("status").innerText = "Audio uploaded successfully.";
    } catch (error) {
        document.getElementById('errorMessage').innerText = "Error: " + error.message;
        document.getElementById("loadingMessage").style.display = 'none';
    }
});

// Handle predicting emotion
document.getElementById('predictEmotionButton').addEventListener('click', async function () {
    const audioFileInput = document.getElementById('audioFile');
    const audioFile = audioFileInput.files[0];

    if (!audioFile) {
        document.getElementById("errorMessage").innerText = "No audio file uploaded for prediction.";
        return;
    }

    const formData = new FormData();
    formData.append("audio", audioFile);

    // Show loading message for prediction
    document.getElementById('loadingMessage').style.display = 'block';
    document.getElementById('loadingMessage').innerText = "Predicting emotion...";

    try {
        // Fetch predictions from the server
        const response = await fetch('/predict_emotion', { // Assuming a different endpoint for predictions
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const result = await response.json();
        displayResults(result);
        document.getElementById("status").innerText = "Predictions fetched successfully.";
        document.getElementById("loadingMessage").style.display = 'none';
        document.getElementById("makeMorePredictionsButton").style.display = 'block'; // Show button to make more predictions
    } catch (error) {
        document.getElementById('errorMessage').innerText = "Prediction Error: " + error.message;
        document.getElementById("loadingMessage").style.display = 'none';
    }
});

// Function to display results
function displayResults(result) {
    const responseDiv = document.getElementById('predictionResult');
    responseDiv.innerHTML = `
        <h2>Predictions:</h2>
        <p><strong>Emotion Probabilities:</strong> ${JSON.stringify(result["Emotion Probabilities"], null, 2)}</p>
        <p><strong>Transcription:</strong> ${result["Transcription"]}</p>
        <p><strong>LLM Interpretation:</strong> ${result["LLM Interpretation"]}</p> <!-- Here -->
    `;
}

// Handle making more predictions
document.getElementById('makeMorePredictionsButton').addEventListener('click', function () {
    // Reset the form and hide elements for a new prediction
    document.getElementById('audioUploadForm').reset();
    document.getElementById('audioPlayer').style.display = 'none';
    document.getElementById('predictEmotionButton').style.display = 'none';
    document.getElementById('makeMorePredictionsButton').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorMessage').innerText = '';
    document.getElementById('predictionResult').innerText = '';
});
