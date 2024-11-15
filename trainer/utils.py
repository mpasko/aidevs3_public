def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f'trainable model parameters: {trainable_model_params}\n \
            all model parameters: {all_model_params} \n \
            percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %'
