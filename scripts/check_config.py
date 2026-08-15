from src import config


def main():
    paths = {
        "MODEL_PATH": config.MODEL_PATH,
        "REPORTS_DIR": config.REPORTS_DIR,
        "HISTORY_CSV_PATH": config.HISTORY_CSV_PATH,
        "DATA_DIR": config.DATA_DIR,
        "LOGS_DIR": config.LOGS_DIR,
    }
    for name, path in paths.items():
        status = "exists" if path.exists() else "missing"
        print(f"{name}: {path} ({status})")
    print("IMG_SIZE:", config.IMG_SIZE)
    print("CLASS_LABELS:", config.CLASS_LABELS)
    print("CONV_LAYER_NAME:", config.CONV_LAYER_NAME)
    print("HEATMAP_INTENSITY:", config.HEATMAP_INTENSITY)


if __name__ == "__main__":
    main()
