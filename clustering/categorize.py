import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

from clustering.kmeans_clustering import KmeansClustering

os.environ['OMP_NUM_THREADS'] = '1' # Avoid physical cores error
# https://stackoverflow.com/questions/77727297/error-with-kmeans-could-not-find-the-number-of-physical-cores-in-windows-7

def get_student_dataset():
    return pd.read_csv("generated_datasets/alumnos.csv")

def get_student_course_dataset():
    return pd.read_csv("generated_datasets/alumnos_cursos.csv")

def get_course_dataset():
    return pd.read_csv("generated_datasets/cursos.csv")

def encoding_dataset():
    df_student = get_student_dataset()
    df_student = df_student.drop(columns=["nombre"])
    df_student_encoding = pd.get_dummies(df_student, columns=["area_interes"])

    df_student_course = get_student_course_dataset()
    df_student_course_grouped = df_student_course.groupby("user_id").agg({
        "rating": "mean",
        "progress": "mean"
    }).reset_index()

    df_student_course_merge = df_student_encoding.merge(df_student_course_grouped, on="user_id", how="left")
    df_student_course_merge.fillna(0, inplace=True)

    return df_student_course_merge

def get_student_courses(user_id):
    student_courses = get_student_course_dataset()
    student_courses = student_courses[student_courses["user_id"] == user_id]
    return student_courses


def get_student_data(user_id):
    students = get_student_dataset()
    student = students[students["user_id"] == user_id]
    return student