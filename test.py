import timm
import timm.optim

# List all optimizers in the timm.optim module
print([attr for attr in dir(timm.optim) if 'optimizer' in attr.lower()])
