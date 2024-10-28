from dataclasses import dataclass

@dataclass
class DataIngestorConfig:
    ARTIFACT_DIR: str
    LOCAL_DIR: str
    DATA_SOURCE: str
    DATA_FILE_PATH: str
    BUCKET_NAME: str
    CLOUD_DIR: str

@dataclass
class TransformConfig():
    BRIGHTNESS: float
    CONTRAST: float
    SATURATION: float
    HUE: float
    RESIZE_SIZE: int
    CENTERCROP_SIZE: int
    RANDOM_ROTATION: int
    NORMALIZE_MEAN: list
    NORMALIZE_STD: list

@dataclass
class DataModuleConfig:
    TRANSFORM: TransformConfig

    BATCH_SIZE: int
    PIN_MEMORY: bool
    TRAIN_DATA_PATH: str
    TEST_DATA_PATH: str
    ARTIFACT_DIR: str
    TRAIN_TRANSFORM_FILE: str
    TEST_TRANSFORM_FILE: str

@dataclass
class ModelConfig:
    NUM_CLASSES: int
    INPUT_SHAPE: list

@dataclass
class ModelModuleConfig:
    ARGS: ModelConfig
    ARTIFACT_DIR: str
    MODEL_PATH: str
    SCRIPTED_MODEL_PATH: str
    STATE_DICT_PATH: str

@dataclass
class TrainerConfig:
    STEP_SIZE: int
    GAMMA: float
    LEARNING_RATE: float
    EPOCHS: int
    ARTIFACT_DIR: str

@dataclass
class EvaluatorConfig:
    ARGS: ModelConfig
    BATCH_SIZE: int
    TEST_DATA_DIR: str
    TEST_TRANFORM_PATH: str
    MODEL_PATH: str
    SCRIPTED_MODEL_PATH: str
    STATE_DICT_PATH: str