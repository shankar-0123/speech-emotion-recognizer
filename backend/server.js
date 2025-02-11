const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 5000;

// Enable CORS and JSON body parsing
app.use(cors());
app.use(express.json());

// Serve static files from the "static" directory
app.use('/static', express.static(path.join(__dirname, 'static')));

// Set up Multer for file uploads
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max file size
});

// Route: Handle file uploads
app.post("/predict-emotion", upload.single("audio"), (req, res) => {
  const file = req.file;

  if (!file) {
    return res.status(400).json({ error: "No file uploaded." });
  }

  const inputPath = path.resolve(file.path);

  // Call the Python script for emotion prediction
  const pythonProcess = spawn("python", ["./scripts/predict_emotion.py", inputPath], {
    env: { ...process.env, TF_ENABLE_ONEDNN_OPTS: "0" },
  });

  let result = "";
  pythonProcess.stdout.on("data", (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error("Python Error:", data.toString());
  });

  pythonProcess.on("close", (code) => {
    // Clean up files
    fs.unlinkSync(inputPath); // Remove the uploaded file

    if (code !== 0) {
      return res.status(500).json({ error: "Failed to predict emotion." });
    }

    try {
      const trimmedResult = result.trim();
      console.log("Python Output:", trimmedResult); // Log the raw output for debugging
      const prediction = JSON.parse(trimmedResult);
      res.json(prediction);
    } catch (error) {
      console.error("JSON Parse Error:", error);
      console.error("Python Output:", trimmedResult); // Log the raw output for debugging
      res.status(500).json({ error: "Invalid prediction response." });
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
