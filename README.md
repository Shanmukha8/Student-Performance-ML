# 🎓 Student Performance Prediction

A beginner-friendly Machine Learning project that predicts student exam scores based on study hours and attendance percentage using Linear Regression.

## 📊 Project Overview

This project demonstrates how machine learning can predict student academic performance. The model analyzes the relationship between daily study hours, attendance percentage, and final exam scores, achieving approximately **85% accuracy**.

## ✨ Features

- **Simple and Clean Code**: Beginner-friendly with detailed comments
- **Linear Regression Model**: Uses scikit-learn's Linear Regression algorithm
- **High Accuracy**: Achieves ~85% R² score
- **Visual Analysis**: 4 comprehensive charts showing:
  - Study Hours vs Performance
  - Attendance vs Performance
  - Prediction Accuracy Plot
  - Error Distribution
- **Sample Predictions**: Tests model with 3 different student profiles
- **Interview-Ready**: Includes key takeaways for technical interviews

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy**: For numerical operations
- **Pandas**: For data handling
- **Scikit-learn**: For machine learning model
- **Matplotlib**: For data visualization

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## 🚀 Usage

Simply run the Python script:
```bash
python student_performance.py
```

The program will:
1. ✅ Create a dataset of 150 students
2. ✅ Split data into training (80%) and testing (20%) sets
3. ✅ Train the Linear Regression model
4. ✅ Display accuracy metrics
5. ✅ Show 4 visualization charts
6. ✅ Make predictions for sample students

## 📈 Model Performance

**Accuracy Metrics:**
- R² Score: ~85% (Excellent accuracy)
- RMSE: ~5 points average error

**Model Formula:**
```
Performance Score = Intercept + (Coefficient × Study_Hours) + (Coefficient × Attendance)
```

## 📊 Sample Output

```
Student 1:
  Study Hours: 2 hours/day
  Attendance: 50%
  → Predicted Score: 45.3/100
  → Grade: C (Needs Improvement)

Student 2:
  Study Hours: 5 hours/day
  Attendance: 75%
  → Grade: B (Good)

Student 3:
  Study Hours: 9 hours/day
  Attendance: 95%
  → Predicted Score: 87.5/100
  → Grade: A (Excellent!)
```

## 📁 Project Structure

```
student-performance-prediction/
│
├── student_performance.py    # Main Python script
├── requirements.txt          # Required packages
└── README.md                # Project documentation
```

## 🔍 Key Insights

1. **Study hours** have a stronger impact on performance than attendance
2. Both features positively correlate with exam scores
3. The model explains ~85% of variance in student performance
4. Model can help identify students who need extra support

## 💡 Interview Talking Points

**Q: What does this project do?**
- Predicts student exam scores using study hours and attendance

**Q: What algorithm did you use?**
- Linear Regression (supervised learning algorithm)

**Q: How accurate is your model?**
- Approximately 85% accuracy (R² score)

**Q: What libraries did you use?**
- pandas (data handling), scikit-learn (ML), matplotlib (visualization), numpy (numerical operations)

**Q: What did the model learn?**
- More study hours → Higher scores
- Better attendance → Higher scores

## 🎯 Future Enhancements

- Add more features (previous grades, socioeconomic factors)
- Try other algorithms (Random Forest, Decision Trees)
- Build a web interface using Flask/Streamlit
- Add real student dataset
- Implement feature scaling and normalization
- Create a prediction API

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or feedback, feel free to reach out or open an issue on GitHub.
