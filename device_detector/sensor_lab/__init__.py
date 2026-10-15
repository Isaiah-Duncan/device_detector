"""Sensor Lab: measure what a laptop can and cannot sense in its own room.

Everything runs locally. Audio is analysed in memory and never saved. Camera frames are analysed in memory and
never saved. Bluetooth addresses are hashed at once. Only numbers (counts, levels, angles) are written to the
results folder. Simulated runs are labeled SYNTHETIC SIGNAL in every output.
"""
SYNTHETIC_LABEL = "SYNTHETIC SIGNAL"
