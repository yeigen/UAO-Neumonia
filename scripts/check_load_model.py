from src.config import CONV_LAYER_NAME
from src.load_model import model_fun


def main():
    model = model_fun()
    model.summary()
    print("input shape:", model.input_shape)
    print("output shape:", model.output_shape)
    layer_names = [layer.name for layer in model.layers]
    print(f"{CONV_LAYER_NAME} present:", CONV_LAYER_NAME in layer_names)


if __name__ == "__main__":
    main()
