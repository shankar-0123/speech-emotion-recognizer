import { useDropzone } from 'react-dropzone';
import { ReactMediaRecorder } from "react-media-recorder";
import { useState } from 'react';
import axios from 'axios';
import inputfileLogo from '../assets/inputfile.svg';
import { Chart } from 'chart.js';  // Changed import
import { convertToWav } from '../utils/audioUtils';

function UploadPage() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [recordedFile, setRecordedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [melSpectrogramBase64, setMelSpectrogramBase64] = useState(null); // State for mel spectrogram base64
  const [polarPlotBase64, setPolarPlotBase64] = useState(null); // State for polar plot base64

  const backendBaseUrl = import.meta.env.VITE_BACKEND_URL; // Use environment variable for backend URL

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      setUploadedFileName(file.name);
      setRecordedFile(null);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: '.wav,.mp3',
    maxSize: 10485760,
    disabled: !!uploadedFile,
  });

  const handleRecordedFile = async (blob) => {
    try {
      const wavFile = await convertToWav(blob);
      setRecordedFile(wavFile);
      setUploadedFileName("");
      setUploadedFile(null);
    } catch (error) {
      console.error('Error converting audio:', error);
      alert('Failed to convert audio recording. Please try again.');
    }
  };

  const handlePrediction = async () => {
    const audioFile = uploadedFile || recordedFile;
    if (!audioFile) {
      alert("Please upload or record an audio file first.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", audioFile);

    try {
      setLoading(true);
      const response = await axios.post(`${backendBaseUrl}/predict-emotion`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data);
      setMelSpectrogramBase64(response.data.melSpectrogramBase64);
      setPolarPlotBase64(response.data.polarPlotBase64);

    } catch (error) {
      console.error("Error predicting emotion:", error);
      alert("Failed to predict emotion. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleStartRecording = (startRecording, stopRecording) => {
    setIsRecording(true);
    startRecording();
    setTimeout(() => {
      handleStopRecording(stopRecording);
    }, 10000);
  };

  const handleStopRecording = (stopRecording) => {
    setIsRecording(false);
    stopRecording();
  };

  return (
    <div className="upload-page">
      <h2>Upload Audio File</h2>
      <div {...getRootProps({ className: 'upload-zone' })}>
        <input {...getInputProps()} />
        <p style={{ color: uploadedFile ? 'gray' : 'black' }}>
          <img src={inputfileLogo} alt='' style={{ marginRight: '10px' }} />
          {uploadedFile ? "" : "Click to upload or drag and drop WAV, MP3 up to 10MB"}
        </p>
        {uploadedFileName && <p>Uploaded File: {uploadedFileName}</p>}
      </div>
      <ReactMediaRecorder
        audio
        onStop={(blobUrl, blob) => handleRecordedFile(blob)}
        render={({ startRecording, stopRecording }) => (
          <div>
            <button
              onClick={() => handleStartRecording(startRecording, stopRecording)}
              className={`record-button ${isRecording ? 'recording' : ''}`}
            >
              {isRecording ? '◉ Recording' : '⬤ Start Recording'}
            </button>
            <button
              onClick={() => handleStopRecording(stopRecording)}
              className="stop-button"
              disabled={!isRecording}
            >
              🚫 Stop Recording
            </button>
          </div>
        )}
      />
      <div className="preview">
        <h3>Preview Audio</h3>
        <audio controls src={uploadedFile ? URL.createObjectURL(uploadedFile) : recordedFile && URL.createObjectURL(recordedFile)}>
          Your browser does not support the audio element.
        </audio>
      </div>
      <button onClick={handlePrediction} className="predict-button" disabled={loading}>
        {loading ? "Predicting..." : "Predict Emotion"}
      </button>
      {prediction && (
        <div className="prediction-result">
          <h3>
            Predicted Emotion: {prediction.emotion.toUpperCase()} 
            {prediction.probabilities && ` (${prediction.probabilities[prediction.emotion].toFixed(2)}%)`}
          </h3>

          {/* Add probabilities table */}
          <div className="probabilities-table">
            <h4>All Emotion Probabilities</h4>
            <table>
              <thead>
                <tr>
                  <th>Emotion</th>
                  <th>Probability</th>
                </tr>
              </thead>
              <tbody>
                {prediction.probabilities && 
                  Object.entries(prediction.probabilities)
                    .sort(([, a], [, b]) => b - a) // Sort by probability in descending order
                    .map(([emotion, probability]) => (
                      <tr key={emotion} className={emotion === prediction.emotion ? 'predicted-emotion' : ''}>
                        <td>{emotion.toUpperCase()}</td>
                        <td>{probability.toFixed(2)}%</td>
                      </tr>
                    ))
                }
              </tbody>
            </table>
          </div>

          <div className="plots-container">
            {melSpectrogramBase64 && (
              <div className="plot-item">
                <h4>Mel Spectrogram</h4>
                <div className="mel-spectrogram">
                  <img src={`data:image/png;base64,${melSpectrogramBase64}`} alt="Mel Spectrogram" />
                </div>
              </div>
            )}
            {polarPlotBase64 && (
              <div className="plot-item">
                <h4>Emotion Probabilities</h4>
                <div className="emotion-chart">
                  <img src={`data:image/png;base64,${polarPlotBase64}`} alt="Emotion Probabilities" />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadPage;
