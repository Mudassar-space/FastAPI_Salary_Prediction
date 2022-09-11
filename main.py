from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel


class Category(BaseModel):

    experience: int


class ResponsePrediction(BaseModel):
    experience: int
    salary: int


df = pd.read_csv("Salary_Data.csv")

app = FastAPI()

x = df[['YearsExperience']]

y = df[['Salary']]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0)


model = LinearRegression()

model.fit(X_train, y_train)


@app.post("/Salary prediction/", tags=["prediction"], status_code=201)
def predict_salary(request: Category):

    print(type(request), "****************")

    raw = request.experience
    print(raw, "????????????????")

    predictions = model.predict([[raw]])

    print(predictions, "<><><><><>><>><")
    data = {"experience": raw, "salary": predictions}
    response = ResponsePrediction(**data)
    # print("<><><><><>",type(data))
    return response


# predictions = model.predict([[12]])

# print(predictions)
