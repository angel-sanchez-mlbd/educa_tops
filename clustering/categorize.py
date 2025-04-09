import pandas as pd
import os

os.environ['OMP_NUM_THREADS'] = '1' # Avoid physical cores error

# Extraer dataset de alumnos.csv
def get_student_dataset():
    return pd.read_csv("generated_datasets/alumnos.csv")

# Extraer dataset de alumnos_cursos.csv
def get_student_course_dataset():
    return pd.read_csv("generated_datasets/alumnos_cursos.csv")

# Extraer dataset de cursos.csv
def get_course_dataset():
    return pd.read_csv("generated_datasets/cursos.csv")

# Extraer dataset de alumnos.csv y alumnos_cursos.csv para mergear agrupando por user_id
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

# Extraer alumnos_cursos.csv para un usuario concreto
def get_student_courses(user_id):
    student_courses = get_student_course_dataset()
    student_courses = student_courses[student_courses["user_id"] == user_id]
    return student_courses

# Extraer cursos.csv y mergear con alumnos_cursos.csv para un usuario concreto
def get_student_courses_detail(user_id):
    df_cursos = get_course_dataset()
    df_alumnos_cursos = get_student_course_dataset()

    alumno_cursos = df_alumnos_cursos[df_alumnos_cursos["user_id"] == user_id]
    alumno_cursos_detail = alumno_cursos.merge(df_cursos, left_on="curso_id", right_on="curso_id", how="left")

    return alumno_cursos_detail

# Extraer alumnos.csv para un usuario concreto
def get_student_data(user_id):
    students = get_student_dataset()
    student = students[students["user_id"] == user_id]
    return student