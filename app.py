from flask import Flask, request, render_template
import os
import pandas as pd

from clustering import categorize
from clustering.kmeans_clustering import KmeansClustering
from clustering.predict_recomendations import PredictRecomendations

users = pd.read_csv("./generated_datasets/alumnos.csv")
app = Flask(__name__, template_folder=os.path.join('form', 'templates'))

@app.route("/")
def home():
    users_id = users["user_id"]
    return render_template('index.html', users=users_id)

@app.route('/student', methods=['POST'])
def student_home():
    user_id = request.form["input_user"] # Obtener id de alumno del formulario
    user = categorize.get_student_data(user_id).to_dict(orient="records")[0] # Extraer información del usuario
    student_courses = categorize.get_student_courses_detail(user_id) # Extraer cursos del usuario
    student_courses = student_courses.to_dict(orient="records") # Convertir a diccionario para procesarlo en html

    plot_student_progress = PredictRecomendations.plot_student_progress(user_id) # Generar imagen del gráfico de progreso
    return render_template('student.html', user=user, student_courses=student_courses, student_progress=plot_student_progress)

@app.route("/recomendations", methods=["POST"])
def recomend():
    user_id = request.form["input_user"] # Obtener id de alumno del formulario
    kmeans_clustering = KmeansClustering()
    df_encoded = categorize.encoding_dataset() # Extraer daos de cursos y alumnos
    df_segmentado = kmeans_clustering.kmeans_apply(df_encoded) # Extraer segmentación por áreas de interés
    df_interacciones = categorize.get_student_course_dataset() # Extraer cursos y alumnos

    top_cursos_usuario = ( # Predecir recomendaciones para el usuario en base a sus áreas de interés
        PredictRecomendations.predict_recommendations_by_user(
            df_segmentado, df_interacciones, user_id)
    )
    courses = top_cursos_usuario.to_dict(orient="records") # Convertir a diccionario para procesarlo en html

    return render_template('recomendations.html', user_id=user_id, courses=courses)

@app.route("/stats")
def show_graph():
    image_graphic_progress = PredictRecomendations.plot_area_progress() # Generar imagen del gráfico de progreso medio
    return render_template("graph_progress.html", area_progress=image_graphic_progress)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)