from flask import Flask, request, render_template
import os
import pandas as pd

from clustering import categorize
from clustering.kmeans_clustering import KmeansClustering
from clustering.predict_recomendations import PredictRecommendations

users = pd.read_csv("./generated_datasets/alumnos.csv")
app = Flask(__name__, template_folder=os.path.join('form', 'templates'))

@app.route("/")
def home():
    users_id = users["user_id"]
    return render_template('index.html', users=users_id)

@app.route('/student', methods=['POST'])
def student_home():
    user_id = request.form["input_user"]
    user = categorize.get_student_data(user_id).to_dict(orient="records")[0]
    student_courses = (categorize.get_student_courses(user_id))
    student_courses = student_courses.to_dict(orient="records")
    print("User student: ", user)
    print("Courses: ", student_courses)
    return render_template('student.html', user=user, student_courses=student_courses)

@app.route("/recomendations", methods=["POST"])
def recommend():
    user_id = request.form["input_user"]
    kmeans_clustering = KmeansClustering()
    df_encoded = categorize.encoding_dataset()
    df_segmentado = kmeans_clustering.kmeans_apply(df_encoded)
    df_interacciones = categorize.get_student_course_dataset()

    recommender = PredictRecommendations(df_segmentado, df_interacciones)
    top_cursos_usuario = recommender.predict_recommendations_by_user(user_id)
    print("Top cursos recomendados: ", top_cursos_usuario)
    courses = top_cursos_usuario.to_dict(orient="records")
    print("Cursos recomendados: ", courses)
    return render_template('recomendations.html', user_id=user_id, courses=courses)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)