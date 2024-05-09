try:
    import torch
    print(torch.cuda.get_device_name())
    print(torch.cuda.is_available())
except ImportError:
    print(" 'torch' module have not been installed.")

# try:
#     import tensorflow as tf
#     tf.test.is_gpu_available()
# except ImportError:
#     print(" 'tensorflow' module have not been installed.")