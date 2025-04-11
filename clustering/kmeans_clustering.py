import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class KmeansClustering:

    # Calcular método del codo
    @staticmethod
    def elbow_method(df_student_course_normalized):
        inertia = []
        k_values = list(range(1, 11))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_student_course_normalized)
            inertia.append(kmeans.inertia_)
        return k_values, inertia

    # Calcular k_optimo en base al método del codo
    def k_optimum(self, df_student_course_normalized):
        k_values, inertia = self.elbow_method(df_student_course_normalized)
        return KneeLocator(k_values, inertia, curve="convex", direction="decreasing").elbow

    # Mostrar método del codo en fráfica
    def plot_elbow_method(self, df_student_course_normalized):
        k_values, inertia = self.elbow_method(df_student_course_normalized)
        plt.plot(k_values, inertia, marker='o')
        plt.xlabel('Número de clústers')
        plt.ylabel('Inercia')
        plt.title('Método del codo')
        plt.show()

    # Aplicar kmeans para la clusterización de alumnos_cursos
    def kmeans_apply(self, df_student_course_normalized):
        df_student_course_anonymized = df_student_course_normalized.drop(columns=["user_id"]) # Anonimizar datos de alumnos

        scaler = StandardScaler()
        df_student_course_scaled = scaler.fit_transform(df_student_course_anonymized) # Transformar datos para poder operar con ellos

        k_optimum = self.k_optimum(df_student_course_scaled) # Calcular k_optimo
        kmeans = KMeans(n_clusters=k_optimum, random_state=42, n_init=100) # Configurar KMeans con 100 repeticiones en base al k_optimo
        df_student_course_normalized['segmento'] = kmeans.fit_predict(df_student_course_anonymized) # Clusterizar alumnos_cursos en segmentos

        return df_student_course_normalized

        # silueta = silhouette_score(df_student_course_normalized, kmeans.labels_)
