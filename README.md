Created a web based attendance system using Hybrid Approach (Haarcascade + LBPH)



- Face Detection: Haar Cascade classifier se camera frame mein faces detect hote hain.
- Face Recognition: LBPH (Local Binary Patterns Histograms) algorithm se detected face ko registered student se match kiya jata hai.
- Hybrid Approach: Haar Cascade detection ke liye aur LBPH recognition ke liye, dono ko combine kiya gaya hai.
- Student Registration: Name, unique roll number aur class ke saath student register hota hai.
- Training Images Capture: Webcam se har student ki 20+ images capture hoti hain, aur blur aur brightness quality check bhi hota hai.
- Live Attendance: Same student 3 consecutive frames mein recognise hone par hi attendance mark hoti hai, jisse galat entry kam hoti hai.
- One Entry Per Day: SQLite database mein ek student ki ek din mein sirf ek attendance record hoti hai.
- Unknown Face Handling: Agar LBPH distance threshold se zyada ho, to face "Unknown" mark hota hai.
- Dashboard & Records: Web dashboard mein students list aur attendance records dekh sakte hain.
- Evaluation Module: Individual aur group images par accuracy, precision, recall aur F1-score nikalne ka system.
- CSV Export: Evaluation results CSV files mein export hote hain.
- Privacy & Security: CSRF protection, secure image upload aur local-only (127.0.0.1) server.

 Tech Stack

| Part | Technology |
|------|-----------|
| Backend | Python 3.11, Flask |
| Computer Vision | OpenCV (Haar Cascade, LBPH) |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Evaluation | scikit-learn, pandas, numpy |
| Testing | pytest |



 Limitations

- Lighting, face angle aur expression ka accuracy par asar padta hai.
- Liveness check nahi hai, isliye photo ya screen se spoof ho sakta hai.
