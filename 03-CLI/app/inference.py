import fire
import joblib
import pooch
import pandas as pd


class StrokePrediction:
    # TODO: provide docstring
    def __init__(self) -> None:
        url = "https://github.com/ironcladgeek/ml2prod/releases/download/v0.0.0/lr_stroke_prediction_v1.pkl"
        md5 = "655376081dd8ff9dbd93868c410ca169"

        self.model_path = pooch.retrieve(
            url=url, known_hash=f"md5:{md5}", progressbar=True
        )
    
    @staticmethod
    def __normalize_str(val):
        return str(val).lower().replace(" ", "_").replace("-", "_")

    def predict(
        self,
        gender: str,
        age: int,
        hypertension: int,
        heart_disease: int,
        ever_married: str,
        work_type: str,
        residence_type: str,
        avg_glucose_level: float,
        bmi: float,
        smoking_status: str,
    ):
        X_test = pd.DataFrame({
            "gender": [self.__normalize_str(gender)],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [self.__normalize_str(ever_married)],
            "work_type": [self.__normalize_str(work_type)],
            "residence_type": [self.__normalize_str(residence_type)],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "smoking_status": [self.__normalize_str(smoking_status)],
        })
        pipeline = joblib.load(self.model_path)
        y_pred = pipeline.predict(X_test)
        return y_pred.item()


if __name__ == "__main__":
    fire.Fire(StrokePrediction)