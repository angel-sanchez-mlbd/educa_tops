import base64
import io

from clustering import categorize
import matplotlib.pyplot as plt
import pandas as pd

class PredictRecomendations:

    # Predecir recomendaciones por usuario
    @staticmethod
    def predict_recommendations_by_user(df_segmented, df_user_courses, user_id, quantity=5):
        user = df_segmented[df_segmented["user_id"] == user_id] # Extraer user_id del dataframe segmentado
        df_clustered_interactions = ( # Mergear dataframe segmentado en alumnos_cursos
            df_user_courses.merge(
                df_segmented[["user_id", "segmento"]],on="user_id",how="left"))
        actual_courses = ( # Extraer alumnos_cursos para el usuario actual
            df_user_courses)[df_user_courses["user_id"] == user_id]["curso_id"]

        segment = user["segmento"].values[0] # Extraer el segmento al que pertenece el alumno
        cluster_data = df_clustered_interactions[df_clustered_interactions["segmento"] == segment] # Extraer alumnos_cursos en base a su segmento

        recomendations = cluster_data.groupby("curso_id").agg({ # Agregar cursos agrupados por el id del curso
            "rating": "mean",
            "progress": "mean",
            "user_id": "count"
        }).rename(columns={"user_id": "num_interactions"}).reset_index() # Cambiar user_id por interacciones de usuarios

        recomendations = recomendations.query("curso_id not in @actual_courses") # Excluir cursos actuales del usuario

        recomendations = recomendations.sort_values( # Ordenar cursos por valoraciones, progreso y número de interacciones
            by=["rating", "progress", "num_interactions"],
            ascending=[False, False, False]
        ).head(quantity)

        # Añadir detalles de cursos (categoría, áreas de interés, etc)
        courses_detail = categorize.get_course_dataset()
        recomendations_detail = recomendations.merge(courses_detail, left_on="curso_id", right_on="curso_id", how="left")

        return recomendations_detail

    @staticmethod
    def plot_area_progress():
        # Obtener datos
        df_alumnos = categorize.get_student_dataset()
        df_cursos = categorize.get_student_course_dataset()

        # Unir datasets por user_id para almacenar progreso y anonimizar dataframe
        df = pd.merge(df_alumnos, df_cursos, on="user_id")
        area_interes = df.drop(columns=["user_id"])

        # Agrupar por área de interés y calcular progreso promedio
        area_progress = area_interes.groupby("area_interes")["progress"].mean().sort_values()

        # Dibujar gráfico
        plt.figure(figsize=(6, 4))
        area_progress.plot(kind='barh', stacked=True, colormap='PuOr')
        plt.xlabel("Progreso medio (%)")
        plt.ylabel("Área de interés")
        plt.title("Progreso promedio por área de interés")
        plt.tight_layout()

        # Convertir a imagen base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Devolvemos gráfico listo para ser mostrado en html
        return plot_image

    @staticmethod
    def plot_student_progress(user_id):
        # Cargar dataframes
        df_cursos = categorize.get_course_dataset()
        df_alumnos_cursos = categorize.get_student_course_dataset()

        # Filtrar solo los cursos del alumno
        alumno_cursos = df_alumnos_cursos[df_alumnos_cursos["user_id"] == user_id]

        # Unir con dataframe de cursos para obtener áreas de interés
        area_cursos = alumno_cursos.merge(df_cursos, left_on="curso_id", right_on="curso_id", how="left")

        # Agrupar por área de interés y calcular progreso promedio
        progreso_alumno = area_cursos.groupby("area_interes")["progress"].mean().sort_values()

        # Dibujar gráfico
        plt.figure(figsize=(6, 3))
        progreso_alumno.plot(kind='bar', stacked=True, colormap='Accent')
        plt.xlabel("Área de interés")
        plt.ylabel("Progreso medio (%)")
        plt.title("Progreso medio por área de interés")
        plt.ylim(0, 100)  # Mostrar valores de 0 a 100%
        plt.yticks(range(0, 101, 30))  # Ticks de 0 a 100 cada 10%
        plt.tight_layout()

        # Convertir a imagen base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Devolver gráfico listo para ser mostrado en html
        return plot_image