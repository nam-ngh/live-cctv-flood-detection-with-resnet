from service_inference.config import S, DRIVER_CHROME_OPTIONS, URL_ABBEY_ROAD, MODEL_PATH
import service_inference.src.m3u8_parser as mps
import service_inference.src.predict as prd

import time
from typing import Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class WebDriverInitError(Exception):
    pass

def init_web_driver(options_config: Dict):
    try:
        options = Options()
        for arg in options_config['arguments']:
            options.add_argument(arg)
        if 'capabilities' in options_config.keys():
            for k, v in options_config['capabilities'].items():
                options.set_capability(k, v)

        return webdriver.Chrome(options=options)
    except Exception as e:
        raise WebDriverInitError(f'Error init. selenium web driver: {e}')
    
def main(url: str) -> None:
    try:
        driver = init_web_driver(options_config=DRIVER_CHROME_OPTIONS)
        model = prd.load_flood_model(model_path=MODEL_PATH,)
        error_count = 0
        while error_count < 30:
            try:
                master_url = mps.get_latest_m3u8_url(
                    driver=driver,
                    target_url=url,
                    wait_time=0.25
                )
                if master_url:
                    segments = mps.get_latest_playlist(master_url)
                    S.logger.info(
                        f'Successfully obtained {len(segments)} video segments. Extracting last frame for prediction'
                    )
                    frame = mps.get_last_frame(segments[-1])
                    cropped_frame = prd.crop_square_centre(frame)
                    pred, conf, prob = prd.predict_flood(
                        model=model,
                        frame=cropped_frame,
                    )
                    prd.plot_prediction(cropped_frame, pred, conf, prob)

                    error_count = 0
                else:
                    raise mps.GetMainUrlError('No m3u8 URL obtained')
            except Exception as e:
                error_count += 1
                S.logger.info(f'Error occured in main: {e}')
                S.logger.info(f'Retrying attempt: {error_count}')
            finally:
                time.sleep(6)
                driver.get_log("performance") # clearing logs
    except WebDriverInitError as e:
        S.logger.info(f'Fatal Error: {e}')
    finally:
        if driver:
            driver.quit()
            S.logger.info("WebDriver closed. Pipeline terminated.")

if __name__ == '__main__':
    main(url=URL_ABBEY_ROAD)