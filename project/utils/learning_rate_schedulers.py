def step_decay_scheduler_generator(initial_lr, coef, epoch_threshold):
    return lambda epoch: initial_lr * (coef ** (epoch // epoch_threshold))