# Nearby Device Detection System

A Windows desktop program that listens for Bluetooth Low Energy (BLE) advertisements and shows what they say. It runs only on your computer. It sends no data to any network.

## What the program can and cannot do

The program can:
- Show each BLE device that advertises near the computer, with a temporary identifier, signal strength (RSSI in dBm), advertisement rate, first-seen time and last-seen time.
- Show advertised service UUIDs, the advertised name (hidden by default) and the manufacturer company number.
- Give a device category with a confidence level and the reasons for it.
- Show events: discovered, not heard, heard again, and reclassified.
- Compare two recording periods (for example "baseline" and "test device on") and export the result.
- Show boxes around phones and laptops in the webcam image, only after you press Start camera and load a local model.

The program cannot:
- Find Bluetooth Classic devices. Many headsets and speakers in an active audio connection are Classic devices. A BLE scan does not see them.
- Find devices that do not advertise (a phone that is idle, or a headset that is connected to another device).
- Find Wi-Fi, cellular, NFC or wired devices.
- Read, decrypt or intercept any communication.
- Name an owner or identify a person.
- Measure distance. RSSI is not a distance (see below).

**Not seeing a device never proves that it is absent.** Seeing a new identifier never proves that a device entered the room.

## Start

1. Install Python 3.11 or newer for Windows.
2. In a Command Prompt in this folder, run:
   `py -3.11 -m venv .venv`
   `.venv\Scripts\activate`
   `pip install -r requirements.txt`
3. Start the program: `python -m device_detector`
4. Choose the source:
   - **Bluetooth LE (real hardware)** uses the Bluetooth adapter.
   - **SYNTHETIC SIGNAL: demo devices** makes simulated devices. A banner and a label on each row and event show that the data is simulated.
5. Press **Start scan**. Press **Stop scan** to stop.

If Bluetooth is off, or Windows denies access, the program shows the reason in panel E and does not start the scan. Open Windows Settings, Bluetooth and devices, and switch Bluetooth on. For permission problems, open Privacy and security, Bluetooth, and allow desktop apps.

## The dashboard

- **A. Nearby devices.** One row for each temporary identifier. Click a row to select it. Point at the Category cell to see the reasons.
- **B. Signal strength.** RSSI of the selected device for the last 70 seconds, with the rolling average. Below the chart: current, minimum, maximum, rolling average, variability (standard deviation), time since the last advertisement, and advertisements per second. An experimental distance guess is shown with a wide range and a warning.
- **C. Device categories.** The number of present identifiers in each category, including Unknown.
- **D. Event timeline.** Time-stamped events. Simulated events start with [SYNTHETIC].
- **E. Sensor status.** Bluetooth, scan permission, webcam, object model, and the list of unsupported abilities.

Confidence colors: green is medium or high, amber is low, grey is none.

## Why RSSI is not a distance

RSSI depends on the transmit power of the device (often unknown), the antennas of both devices, the way the device is held, the body of the person, walls, furniture, reflections that add or cancel the signal, interference on the 2.4 GHz band, and the way the adapter and driver report the value. Two identical devices at the same distance can differ by 10 dB or more. Use the trend of one device in one place. Do not use the absolute number as a distance. The program can show a distance guess, which is always labeled EXPERIMENTAL and can be wrong by a large factor.

## How device categories are decided

The program uses only information that the device advertises:
- Standard service UUIDs (for example Heart Rate gives Fitness device).
- The advertised appearance value (for example Watch).
- The advertised name (weak evidence, because owners and vendors choose names freely).

A manufacturer company number alone never gives a category. A device with only a company number is "Unknown device". When two kinds of evidence point to different categories, the confidence is lowered and the reason says so.

Confidence levels: none, low, medium, high. The reason list explains each one.

## Randomized addresses

Many phones and accessories change their Bluetooth address every few minutes. Each new address appears as a new device. Because of this, the number of identifiers is an upper bound on the number of real devices. Panel C shows the number of new identifiers per minute. A high number with few real devices points to address rotation.

## Experiment mode

Use the **Experiment** tab.
1. Press **Start** with the name "baseline". Let the program listen for a period (for example 2 minutes). Press **Stop**.
2. Bring a permitted test device (your own device, or a device whose owner agreed) near the computer. Turn it on or wake it.
3. Record a second segment (for example "test"). Press **Stop**.
4. Choose the two segments and press **Compare**.
5. Press **Save comparison** to write JSON. Use the toolbar buttons to export events as CSV or JSON.

The comparison shows new identifiers, identifiers not seen again, and changes in advertisement rate and mean RSSI. The text explains the difference between a device becoming discoverable and a device entering the room.

## Webcam detection (optional)

The camera is off until you press **Start camera**. You must tick the consent box. The program draws boxes and confidence values on phones and laptops. It does not detect faces, does not identify people, and does not record or save video or pictures.

The program needs a local YOLOv8 ONNX model (80 COCO classes, for example `yolov8n.onnx`). The model is not included. See **Limits and privacy** tab for export instructions. The COCO class list has "cell phone" and "laptop" but no "tablet". A tablet can be shown as one of these or not at all.

## Privacy

- All processing is local. The program has no network code.
- Real Bluetooth addresses are hashed with a random key made at each start. The key stays in memory. Identifiers cannot be traced back to an address after you close the program.
- Raw advertisement payloads are not stored. Manufacturer data is reduced to the company number.
- Advertised names are hidden by default and are saved only if you tick **Show advertised names**.
- Data stays in memory unless you tick **Save to database file**. **Clear all data** removes everything. Retention of saved data is your choice (see `Settings.retention_days` in `device_detector/config.py`, and **Clear all data**).
- The program never connects to a device and never scans in the background. It scans only between Start and Stop.

Use this program only in places and on devices where you have permission. Do not use it to track a person.

## Tests

`python -m pytest -q tests`

The tests use simulated data and fake Bluetooth objects. They check parsing, classification, RSSI statistics, tracking, address rotation, disappearance and rediscovery, exports, experiments, error handling when Bluetooth or Bleak is missing, the YOLO decoder, and the dashboard (offscreen). **They do not prove that the program works with real Bluetooth hardware on Windows.** See `WINDOWS_TESTS.md`.

## Build a Windows program

Run `build_exe.bat`. It makes a virtual environment, installs the packages, runs the tests, and runs PyInstaller with `device_detector.spec`. The result is `dist\DeviceDetector\DeviceDetector.exe`. Copy the whole `DeviceDetector` folder to move it. If you use the camera, keep the `.onnx` model in a known location and choose it in the Camera tab.

## Files

- `device_detector/adparse.py` parses advertisement data.
- `device_detector/scanner.py` runs the Bleak scan in a background thread.
- `device_detector/rssi.py` makes the signal statistics and the experimental distance.
- `device_detector/classifier.py` gives categories, confidence and reasons.
- `device_detector/tracker.py` tracks devices and makes events.
- `device_detector/store.py` saves events and devices (SQLite) and exports CSV or JSON.
- `device_detector/experiment.py` records and compares segments.
- `device_detector/vision.py` runs the optional ONNX object detector.
- `device_detector/synthetic.py` makes the simulated devices.
- `device_detector/ui/main_window.py` is the dashboard.

## Sensor Lab (benchmark tool)

`device_detector/sensor_lab` measures what a laptop can sense in its own room: program cost, microphone channels, sound direction against camera position, Bluetooth signal against distance, and false alarms in a quiet room. See `SENSOR_LAB.md`. Start it with `python -m device_detector.sensor_lab`. Build it with `build_sensor_lab.bat`.
