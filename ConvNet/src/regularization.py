
def l1_regularizer(parameters):
    l1_lambda = 0.01
    l1_norm = 0.0

    for param in parameters:
        l1_norm += param.abs().sum()

    return l1_lambda * l1_norm

def l2_regularizer(parameters):
    l2_lambda = 0.01
    l2_norm = 0.0

    for param in parameters:
        l2_norm += param.pow(2).sum()

    return l2_lambda * l2_norm

def elastic_regularizer(parameters):
    elastic_lambda = 0.01
    elastic_alpha = 0.1

    elastic_norm = 0.0
    for param in parameters:
        l1_and_l2 = (1 - elastic_alpha) * param.pow(2).sum() + elastic_alpha * param.abs().sum()
        elastic_norm += l1_and_l2

    return elastic_lambda * elastic_norm

