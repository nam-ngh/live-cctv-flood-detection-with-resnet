import os
import cv2
import json
import time
import m3u8
import requests
import tempfile
from loguru import logger

class GetMainUrlError(Exception):
    pass

class GetPlaylistError(Exception):
    pass

class SegmentDownloadError(Exception):
    pass

def get_latest_m3u8_url(driver, target_url, wait_time=5):
    """
    :params driver: Selenium web driver object
    :params target_url: The URL of the live stream page.
    :params wait_time: Time to wait (in seconds) for HLS requests to load.
    :returns The captured master M3U8 URL (string) or None.
    """
    try:
        # navigate and wait
        logger.info(f"Navigating to: {target_url}")
        driver.get(target_url)
        logger.info(f"Waiting {wait_time}s for network requests...")
        time.sleep(wait_time)

        # retrieve and parse network logs
        logger.info("Retrieving and parsing performance logs...")
        logs = driver.get_log("performance")
        
        main_m3u8_url = None
        
        for i, log in enumerate(logs):
            try:
                # Load the JSON message from the log entry
                message = json.loads(log['message'])['message']
            except (json.JSONDecodeError, KeyError):
                continue # Skip logs that are not correctly formatted

            # Look for the event where a network request is initiated
            if message['method'] == 'Network.requestWillBeSent':
                url = message['params']['request']['url']

                # Filter for the HLS master playlist file
                if '.m3u8' in url:
                    # Found the master M3U8 URL with the expiring tokens
                    main_m3u8_url = url
                    logger.info(f"M3U8 URL captured at log msg. {i}!")
        return main_m3u8_url

    except Exception as e:
        raise GetMainUrlError(f'Error fetching main m3u8 URL: {e}')

def get_latest_playlist(m3u8_url):
    """Get .m3u8 playlist and returns list of .ts segment URLs."""
    try:
        response = requests.get(m3u8_url)
        response.raise_for_status() # Raise exception for bad status codes
        # parse the playlist
        playlist = m3u8.loads(response.text)
        # extract the full URLs for ts segments
        base_url = m3u8_url.rsplit('/', 1)[0] + '/'
        segment_urls = [
            # Ensure the segment URL is absolute
            base_url + segment.uri if not segment.uri.startswith('http') else segment.uri
            for segment in playlist.segments
        ]
        return segment_urls
    except Exception as e:
        raise GetPlaylistError(f'Error getting video segment playlist: {e}')
    
def get_last_frame(ts_url):
    """Downloads the .ts file and extracts the last frame as a NumPy array."""
    temp_file = None
    cap = None
    
    try:
        # Download the video file
        logger.info(f"Downloading {ts_url}...")
        response = requests.get(ts_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Use tempfile for better security and automatic cleanup
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            temp_file = f.name
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.debug(f"Downloaded to {temp_file}")
        
        # open video with opencv
        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            raise Exception(f"Cannot open video file {temp_file}")
        
        # total frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            raise Exception(f"Invalid frame count: {frame_count}")
        
        logger.debug(f"Video has {frame_count} frames")
        
        # set position to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        
        # Read the last frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            raise Exception("Could not read the last frame")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.success(f'Frame extracted: {frame_rgb.shape}')
        return frame_rgb
    
    except requests.exceptions.RequestException as e:
        raise SegmentDownloadError(f"Error downloading {ts_url}: {e}")
    
    finally:
        # cleanup
        if cap is not None:
            cap.release()
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")