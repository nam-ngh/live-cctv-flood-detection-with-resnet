from utils.system import SysUtils

S = SysUtils(log_file_path='logs/training.log')

FLOOD_IMAGES_PATH=S.from_env('FLOOD_IMAGES_PATH')
DRY_IMAGES_PATH=S.from_env('DRY_IMAGES_PATH')

EPOCHS=int(S.from_env('EPOCHS'))
MODEL_PATH = S.from_env('MODEL_PATH')