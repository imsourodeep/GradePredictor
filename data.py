# data.py
# This file creates a fake (synthetic) dataset of 500 students
# We don't need a real dataset — we generate realistic data ourselves!

import pandas as pd
import numpy as np

def generate_data(n=500, save=True):
    np.random.seed(42)  # Makes sure we get the same data every time

    study_hours     = np.random.uniform(0, 10, n)        # 0 to 10 hrs/day
    attendance      = np.random.uniform(40, 100, n)      # 40% to 100%
    sleep_hours     = np.random.uniform(3, 10, n)        # 3 to 10 hrs/night
    prev_score      = np.random.uniform(30, 100, n)      # Previous exam score
    extracurricular = np.random.randint(0, 2, n)         # 0 = No, 1 = Yes

    # Formula to generate final grade (with some randomness/noise)
    # Each factor contributes differently to the final score
    grade = (
        study_hours     * 3.5 +
        attendance      * 0.3 +
        sleep_hours     * 1.5 +
        prev_score      * 0.4 +
        extracurricular * 2.0 +
        np.random.normal(0, 3, n)   # Random noise to make it realistic
    )

    # Clip grade between 0 and 100
    grade = np.clip(grade, 0, 100)

    df = pd.DataFrame({
        "study_hours":      study_hours,
        "attendance":       attendance,
        "sleep_hours":      sleep_hours,
        "prev_score":       prev_score,
        "extracurricular":  extracurricular,
        "final_grade":      grade
    })

    if save:
        df.to_csv("student_data.csv", index=False)
        print(f"✅ Dataset created with {n} students and saved to student_data.csv")

    return df

if __name__ == "__main__":
    df = generate_data()
    print(df.head())
