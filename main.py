import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Experience(BaseModel):
    experience: float


class ResponsePrediction(BaseModel):
    experience: float
    salary: int


class APIResponse(BaseModel):
    status: bool
    message: str


df = pd.read_csv("Salary_Data.csv")

app = FastAPI()


x = df[['YearsExperience']]
y = df[['Salary']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

model = LinearRegression()

model.fit(X_train, y_train)

@app.post("/Salary prediction/", status_code=status.HTTP_201_CREATED, response_model=ResponsePrediction, tags=["prediction"])
# responses={status.HTTP_201_CREATED: {"model": ResponsePrediction}})
# status.HTTP_401_UNAUTHORIZED: {"model": APIResponse},
# status.HTTP_404_NOT_FOUND: {"model": APIResponse},
# status.HTTP_400_BAD_REQUEST: {"model": APIResponse},
# status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": APIResponse}})
def predict_salary(request: Experience):

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
