import tinker


service_client = tinker.ServiceClient()


def main():
    print_models()

    base_model = "meta-llama/Llama-3.2-1B"
    training_client = service_client.create_lora_training_client(base_model=base_model)


def print_models():
    print("Available models:")
    for item in service_client.get_server_capabilities().supported_models:
        print("- " + (item.model_name or ""))


if __name__ == "__main__":
    main()
