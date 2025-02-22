## 🚀 Features
- 📥 **Downloads & processes** the LFW dataset for face recognition.
- 🖼️ **Detects faces** in an input image.
- 🎭 **Compares faces** with known identities from LFW.
- 🖊️ **Draws bounding boxes & face landmarks**.
- 📊 **Displays the results** using Matplotlib.


Required Packages:
face_recognition
opencv-python
numpy
matplotlib
scikit-learn

🔬 How It Works
Loads the LFW dataset and extracts known face encodings.
Detects faces in the input image using face_recognition.
Identifies faces by comparing them with known encodings.
Draws bounding boxes & landmarks around detected faces.
Displays the final image using Matplotlib.

🖼️ Example Output
The script will:

Detect a face in an image.
Identify the person (if they exist in LFW).
Draw a rectangle and label their name.
Mark facial landmarks (eyes, nose, lips, etc.).
Display the output image.


⚡ Issues & Contributions
Found a bug or have suggestions? Feel free to open an issue or submit a pull request.

