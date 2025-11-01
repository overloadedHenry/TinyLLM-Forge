from torch.optim.lr_scheduler import LambdaLR
import math

def get_num_training_steps(num_samples, per_device_train_batch_size, num_epochs, world_size=8, gradient_accumulation_steps=1):
    return math.ceil(num_samples / (per_device_train_batch_size * world_size * gradient_accumulation_steps)) * num_epochs

def get_cosine_schedule_with_lower_bound(optimizer, num_training_steps):

    def lr_lambda(current_step: int):
        return 0.1 + 0.5 * (1 + math.cos(math.pi * current_step / num_training_steps))

    return LambdaLR(optimizer, lr_lambda)

