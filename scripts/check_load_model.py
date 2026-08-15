from src.load_model import model_fun


def main():
    model = model_fun()
    model.summary()
    print("input shape:", model.input_shape)
    print("output shape:", model.output_shape)
    layer_names = [layer.name for layer in model.layers]
    print("conv10_thisone present:", "conv10_thisone" in layer_names)


if __name__ == "__main__":
    main()
