# Sensor Lab

Sensor Lab measures what a laptop can and cannot sense in its own room. It runs five tests, writes the numbers to a spreadsheet file, and writes a plain summary. Use it only in your own space, with your own devices, and with people who agree.

## What it does with your data

- Audio and camera images are analysed in memory and are never saved.
- Bluetooth addresses are hashed at once with a random key and are never saved.
- Only numbers (levels, angles, counts, timings) go into the results folder.
- Simulated runs print and save the label SYNTHETIC SIGNAL. Numbers from a simulated run say nothing about your hardware.

## Start

1. Install the packages: `pip install -r requirements.txt`
2. Try the simulated run first, to see how the output looks: `python -m device_detector.sensor_lab --run all --synthetic`
3. Run the real tests: `python -m device_detector.sensor_lab` and choose a test from the menu. You can also name a test: `--run stereo`, `--run direction`, `--run ble`, `--run cost`, `--run idle`.

Each real run writes a folder `sensor_lab_results\run_<date>_<time>` with three files: `summary.md` (read this), `runs.csv` (open in a spreadsheet; one row per test), and `results.json` (all numbers).

## The five tests

**1. Cost (`cost`).** Runs the camera loop with motion detection for 20 seconds (`--cost-seconds`). Add `--model yolov8n.onnx` to include the object detector. It reports frames per second, time per frame, CPU, memory and temperature (Windows normally gives no temperature to programs).

**2. Microphone channels (`stereo`).** Records 3 seconds while you make noise. It tells you whether the laptop gives the program two different microphone channels. If both channels are identical, Windows or the driver merges the microphones, and test 3 cannot work on this laptop.

**3. Direction (`direction`).** The program asks you to clap at the left, right and center of the camera view while holding a moving object (a hand or a phone screen) there. For each clap it finds the direction from the sound, and the position of motion in the image, and compares both with where you really were. It reports how often each was right (guessing gives about 33%) and how often they agreed. Set `--spacing` to the real distance between your two microphones in meters (about 0.10 for many laptops; the value is a guess unless you check the laptop's documentation). If the report says the channels look swapped, run again with `--swap`. If your camera shows a mirror image, left and right in the picture will be reversed.

**4. Bluetooth distance (`ble`).** Choose your own device (for example earbuds in pairing mode) from the list. The program records its signal strength for 60 seconds (`--seconds-each`) at 1, 2, 4 and 8 meters (`--distances`). It fits a straight line of signal against the logarithm of distance, reports the path-loss exponent, and tells you how wrong a single distance estimate usually is. It also tells you which neighbouring distances overlap and cannot be told apart. Run it a second time with the device in a pocket, to see the effect of your body.

**5. Idle false alarms (`idle`).** Leave the room quiet and still for 30 minutes (`--idle-minutes`). The program counts sound events and motion events. With nothing happening, every event is a false alarm. It also counts how many different Bluetooth identifiers appear, which shows how often nearby devices change address.

**6. Haptic delay (`haptic`, optional, sends real texts).** Measures the delay between a text request and the buzz of your own phone, as heard by the laptop microphone. It sends bursts of texts through your own Live Ear `/api/notify` function, with a long pause between bursts so that each buzz belongs to one burst. A burst of one text (`--per-burst 1`) measures the delay. A burst of two or more texts at the app's gap (`--haptic-gap 3`) shows whether the buzzes stay separate. The test is not part of `--run all`, because each text can cost money on your SMS plan, and the program asks you to confirm before it sends anything.

```
python -m device_detector.sensor_lab --run haptic --notify-url https://your-site.vercel.app/api/notify --bursts 5 --per-burst 2 --haptic-gap 3
```

The program asks for the passcode and does not save it. You can also set it in the `SENSOR_LAB_PASSCODE` environment variable for one session.

Conditions for a fair result: put your own phone flat on the same hard table as the laptop, switch Vibrate on Silent on, and keep the room quiet. The detector listens for a rise in the 100 to 400 Hz band, so a knock on the table, a door or heavy typing can look like a buzz. A phone in a pocket, in a hand or on a soft surface will often not be heard at all. A missed buzz is inconclusive and is not proof that the text did not arrive.

What it cannot do: the laptop cannot see the text message itself, because texts travel over the phone's cellular radio, not over Bluetooth or Wi-Fi. It hears only the buzz.

## How to read the results

- A result applies to this laptop, in this room, on this day.
- A test that finds nothing is inconclusive. It is not proof that nothing is there.
- Sound direction from two microphones a few centimeters apart has a resolution of roughly 8 degrees near the center, so it can tell left from right but cannot find a precise position.
- The distance estimate from Bluetooth signal strength is usually off by a factor of 1.5 or more. The test measures this for your laptop.

## Windows hardware checks still needed

No test in this package has been run on Windows with a real microphone, camera or Bluetooth adapter. A person must do these checks:

1. `python -m pytest -q tests` passes on Windows.
2. `--run stereo` returns "different" or "identical" with no error, and `--run cost` runs with the real camera.
3. In `--run direction`, clapping at the left gives a left result in at least 6 of 10 claps, or the report explains why not.
4. In `--run ble`, the device list shows your own device, and the results table has numbers for each distance.
5. `--run idle --idle-minutes 5` finishes and writes the results folder.
6. `build_sensor_lab.bat` builds `dist\SensorLab\SensorLab.exe`, and the program starts.
7. (Optional) `--run haptic` with your notify address sends the texts, and the report shows how many buzzes were heard and their delay. Check your SMS costs first.
8. Windows microphone and camera privacy settings allow desktop apps. If a test says no microphone was found, check Settings, Privacy and security.
