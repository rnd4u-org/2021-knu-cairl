from split_data import load_data
from build_models import build_model

x_train, y_train, x_test, y_test = load_data()
model = build_model()

model.fit(x_train, y_train)