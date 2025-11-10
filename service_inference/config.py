from utils.system import SysUtils

# set up utility
S = SysUtils(log_file_path='logs/inference.log')

DRIVER_CHROME_OPTIONS = {
    'arguments': [
        "--headless", # run in background - no UI
        "--disable-gpu", # recommended for headless
        "--window-size=1920,1080", # set large viewport
        "--log-level=3",
    ],
    'capabilities': {
        'goog:loggingPrefs': {'performance': 'ALL'}
    }
}

URL_ABBEY_ROAD = S.from_env('URL_ABBEY_ROAD')
MODEL_PATH = S.from_env('MODEL_PATH')