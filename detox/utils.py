def get_gradient_norm(parameters, norm_type=2):
    parameters_with_gradient = [p for p in parameters if p.grad is not None]

    total_norm = 0
    try:
        for p in parameters_with_gradient:
            total_norm += (p.grad.data ** norm_type).sum()
        total_norm = total_norm ** (1.0 / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0
    try:
        for p in parameters:
            total_norm += (p.grad.data ** norm_type).sum()
        total_norm = total_norm ** (1.0 / norm_type)
    except Exception as e:
        print(e)

    return total_norm
