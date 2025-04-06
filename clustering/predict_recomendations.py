class PredictRecommendations:
    def __init__(self, df_segmented, df_user_courses):
        self.df_segmented = df_segmented
        self.df_user_courses = df_user_courses
        self.df_clustered_interactions = self.merge_data()

    def merge_data(self):
        return self.df_user_courses.merge(
            self.df_segmented[["user_id", "segmento"]],
            on="user_id",
            how="left"
        )

    def predict_recommendations_by_user(self, user_id, quantity=5):
        user = self.df_segmented[self.df_segmented["user_id"] == user_id]
        actual_courses = self.df_user_courses[self.df_user_courses["user_id"] == user_id]["curso_id"]

        print("Fila usuario segmento: ", user["segmento"].values)
        segment = user["segmento"].values[0]
        print("Segmento: ", segment)
        cluster_data = self.df_clustered_interactions[self.df_clustered_interactions["segmento"] == segment]

        recomendations = cluster_data.groupby("curso_id").agg({
            "rating": "mean",
            "progress": "mean",
            "user_id": "count"
        }).rename(columns={"user_id": "num_interactions"}).reset_index()

        recomendations = recomendations.query("curso_id not in @actual_courses")

        recomendations = recomendations.sort_values(
            by=["rating", "progress", "num_interactions"],
            ascending=[False, False, False]
        ).head(quantity)

        return recomendations

