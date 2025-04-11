import pandas as pd
import random


# Generar datos simulados para 100 alumnos y 20 cursos
num_users = 50
num_courses = 20

# Crear lista de alumnos y cursos
users = [f"user_{i+1}" for i in range(num_users)]
courses = [f"course_{j+1}" for j in range(num_courses)]

# Crear dataset alumnos_cursos
data = []
for  user in users:
    num_interactions = random.randint(1, 10)  # cada usuario ha interactuado con 1 a 10 cursos
    interacted_courses = random.sample(courses, num_interactions)
    for course in interacted_courses:
        rating = round(random.uniform(1, 5), 1)  # puntuación entre 1 y 5
        progreso = round(random.uniform(0, 100), 2)  # progreso entre 0 y 100
        data.append([user, course, rating, progreso])

# Crear tabla de alumnos
user_data = []
for index, user in enumerate(users):
    name = "Name " + str(index)
    edad = random.randint(18, 60)
    area_interes = random.choice(["Tecnología", "Negocios", "Arte", "Ciencias", "Salud", "Educación"])
    user_data.append([user, name, area_interes])

df_users = pd.DataFrame(user_data, columns=[
    "user_id", "nombre", "area_interes"
])

# Crear tabla de cursos
course_data = []
for course in courses:
    categoria = random.choice(["Programación", "Marketing", "Diseño", "Matemáticas", "Idiomas"])
    area_interes = random.choice(["Tecnología", "Negocios", "Arte", "Ciencias", "Salud", "Educación"])
    dificultad = random.choice(["Básico", "Intermedio", "Avanzado"])
    duracion_horas = random.randint(5, 50)
    valoracion_instructor = round(random.uniform(3.0, 5.0), 1)
    course_data.append([course, categoria, area_interes, dificultad, duracion_horas, valoracion_instructor])

# Crear tabla de alumnos_cursos
df_users_courses = pd.DataFrame(data, columns=[
    "user_id", "curso_id", "rating", "progreso"
])


df_courses = pd.DataFrame(course_data, columns=[
    "curso_id", "categoria", "area_interes", "dificultad", "duracion_horas", "valoracion_instructor"
])

# Crear csv con todos los datos generados
df_users_courses.to_csv("alumnos_cursos.csv", index=True)
df_users.to_csv("alumnos.csv", index=True)
df_courses.to_csv("cursos.csv", index=True)

print("Cursos generados satisfactoriamente.")
