from tensorflow.keras import models
import numpy as np

MODEL_DIR_PATH = "D:/AI/model/ai-system-project/"
MODEL_NAMES = ["training_conv1d.h5", "training_fcnn.h5"]
nploader = np.load("npz/TFSR_n_mfcc_80_splited.npz")
x_test, x_test_2d, y_test = nploader["x_test"], np.expand_dims(nploader["x_test"], axis=-1), nploader["y_test"]
print(np.shape(x_test), np.shape(x_test_2d), np.shape(y_test))

model = models.load_model(filepath=(MODEL_DIR_PATH + "training_conv2d.h5"))
evaluate_result = model.evaluate(
    x=x_test_2d,
    y=y_test,
    verbose=1,
    return_dict=True
)
print(evaluate_result)

for model_name in MODEL_NAMES:
    model = models.load_model(filepath=(MODEL_DIR_PATH + model_name))
    evaluate_result = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=1,
        return_dict=True
    )
    print(evaluate_result)
