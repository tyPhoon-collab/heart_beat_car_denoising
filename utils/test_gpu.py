try:
    import torch

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())
except ImportError:
    print(" 'torch' module have not been installed.")

# try:
#     import tensorflow as tf
#     tf.test.is_gpu_available()
# except ImportError:
#     print(" 'tensorflow' module have not been installed.")
