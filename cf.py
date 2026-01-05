# config.py
class Config:
    # 数据路径
    TRAIN_PATH = 'TrainingData.csv'
    VAL_PATH = 'ValidationData.csv'

    # 模型参数
    INPUT_DIM = 520
    FC1_DROPOUT = 0.5
    DAE_NOISE_STD = 0.02

    # 训练参数
    SEQ_LENGTH = 5
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-2
    FLOOD_LEVEL = 0.3
    TRAIN_SUBSET_RATIO = 1.0

    # GAN参数
    GAN_EPOCHS = 100
    GAN_BATCH_SIZE = 64

    # DAE参数
    DAE_EPOCHS = 30