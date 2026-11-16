import pickle
import uvicorn

from typing import Dict, Any
from fastapi import FastAPI

from typing import Literal, Optional
from pydantic import BaseModel, Field

from pydantic import ConfigDict

class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid")
# categorical
    Name: str
    Sex: Literal["male", "female"]
    Ticket: str
    Cabin: Optional[str] = None                        # mostly missing in the dataset
    Embarked: Optional[Literal["C", "Q", "S"]] = None  # 2 missing in train.csv

    # numerical
    PassengerId: int = Field(ge=1)
    Pclass: Literal[1, 2, 3]
    Age: Optional[float] = Field(default=None, ge=0, le=120)  # ~20% missing
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)


class PredictResponse(BaseModel):
    survival_probability: float
    survived: bool

app = FastAPI(title="titanic-passenger-survival-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(passenger):
    y_pred = pipeline.predict_proba(passenger)[0, 1]
    return float(y_pred)

@app.post("/predict",response_model=PredictResponse)
def predict(passenger: Passenger) -> PredictResponse:
    prob = predict_single(passenger.model_dump())

    return PredictResponse(
        survival_probability=prob,
        survived=prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

