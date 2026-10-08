"""The two dataframes under test. `crashes` is the fake dataset from the lecture's
hallucination-proof script: the model cannot know its numbers from training data."""

import pandas as pd
from seaborn import load_dataset


def titanic() -> pd.DataFrame:
    return load_dataset("titanic")


def crashes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "planet": ["Zorgon"] * 30 + ["Xylpha"] * 20 + ["Brimtak"] * 50,
            "survived_crash": [1] * 18 + [0] * 12 + [1] * 5 + [0] * 15 + [1] * 35 + [0] * 15,
            "crew_role": (
                ["pilot"] * 10 + ["medic"] * 20 + ["pilot"] * 8 + ["medic"] * 12
                + ["pilot"] * 25 + ["medic"] * 25
            ),
            "age": list(range(20, 50)) + list(range(25, 45)) + list(range(18, 68)),
        }
    )


FRAMES = {"titanic": titanic, "crashes": crashes}
SURVIVED_COL = {"titanic": "survived", "crashes": "survived_crash"}
