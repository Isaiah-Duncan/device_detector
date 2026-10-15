# Checks that need real Windows hardware

Status of this release: the code was written and tested on Linux with simulated data and fake Bluetooth objects. **No test on Windows, no test with a real Bluetooth adapter, no test with a real webcam, and no .exe build has been done.** Each check below must be done by a person. Write the result in the last column.

| # | Check | Expected result | Result |
|---|-------|-----------------|--------|
| 1 | Run `python -m pytest -q tests` on Windows | All tests pass | |
| 2 | Start the program, choose the synthetic source, press Start | Yellow SYNTHETIC banner, 6 or 7 rows, events appear, plot moves | |
| 3 | Choose real Bluetooth, press Start with Bluetooth ON | Panel E shows "Scanning". Rows appear for nearby advertisers | |
| 4 | Switch Bluetooth OFF in Windows, press Start | Panel E gives a clear message. No crash. Start is usable again | |
| 5 | Switch Bluetooth off while a scan runs | The scan stops, panel E explains why | |
| 6 | Deny Bluetooth access for desktop apps | Panel E says access is denied and where to change it | |
| 7 | Use a computer with no Bluetooth adapter (or disable it in Device Manager) | Panel E says no adapter was found | |
| 8 | Put your own BLE device (for example a fitness band or earbuds in pairing mode) near the computer | A new row appears. RSSI changes when you move the device closer or farther. Do not expect an exact distance | |
| 9 | Turn the device around in your hand | RSSI changes by several dB with no change in distance. This confirms the RSSI warning | |
| 10 | Switch the test device off, wait 30 seconds | Event "not heard" appears. Switch it on: "heard again" or a new identifier | |
| 11 | Leave a phone near the computer for 20 minutes | Count the new identifiers. Rotation can show as many identifiers for one phone | |
| 12 | Play audio over a Classic Bluetooth headset | The headset may not appear. This is expected | |
| 13 | Run a real scan for 30 minutes in a busy place | The window stays responsive. Memory stays stable. Check the "dropped" count is zero or small | |
| 14 | Export CSV and JSON, open them | Files open. Columns and notice text are correct | |
| 15 | Tick "Save to database file", scan, close, open `device_detector.db` | Names are NULL unless "Show advertised names" was ticked | |
| 16 | Camera tab: press Start camera without the consent box | The camera does not start | |
| 17 | Export YOLOv8n to ONNX (see the Limits tab), load it, start the camera, show a phone and a laptop | Boxes with confidence. Faces get no box and no label | |
| 18 | Press Stop camera | The camera light goes off | |
| 19 | Run `build_exe.bat`, start `dist\DeviceDetector\DeviceDetector.exe` on a clean computer | The window opens. Real scan works (checks 3 and 8 again) | |
| 20 | Check that Windows Firewall shows no network connection from the program | No connection | |

If a check fails, note the exact message in panel E and the Windows version.
