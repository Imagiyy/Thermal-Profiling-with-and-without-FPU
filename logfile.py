#!/usr/bin/env python3
"""
logfile.py - logs STM32F401RE thermal-profiling CSV stream to a file.

Usage:
    python logfile.py /dev/ttyACM0 --baud 115200
    python logfile.py COM5 -o run1.csv

Requires: pip install pyserial
"""

import argparse
import csv
import os
import sys
from datetime import datetime
import serial
import serial.tools.list_ports

HEADER = [
    "timestamp_ms",
    "vin_V",
    "vout_V",
    "current_mA",
    "power_loss_mW",
    "ambient_temp_C",
    "heatsink_temp_C",
    "fault_code",
]


def parse_row(line):
    """Return fields if line is a well-formed data row, else None."""
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != len(HEADER):
        return None
    try:
        int(fields[0])
        for f in fields[1:-1]:
            float(f)
        int(fields[-1], 0)  # Base 0 handles decimal and hex (e.g. 0x00)
    except ValueError:
        return None
    return fields


def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    for p in ports:
        print(f"  {p.device} - {p.description}")


def main():
    ap = argparse.ArgumentParser(description="STM32 thermal dataset CSV logger")
    ap.add_argument("port", nargs="?", default="/dev/ttyACM0", help="serial port (default: /dev/ttyACM1)")
    ap.add_argument("--baud", type=int, default=115200, help="baud rate (default: 115200)")
    ap.add_argument("-o", "--output", help="custom output CSV file name")
    ap.add_argument("--list", action="store_true", help="list available serial ports and exit")
    args = ap.parse_args()

    if args.list:
        list_ports()
        return 0

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        print("\nAvailable ports:")
        list_ports()
        return 1

    filename = args.output or datetime.now().strftime("thermal_dataset_%Y%m%d_%H%M%S.csv")
    print(f"Logging {args.port} @ {args.baud} baud -> {filename}  (Ctrl+C to stop)")
    ser.reset_input_buffer()

    rows = skipped = 0
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            f.flush()

            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                print(line)
                if line.startswith("#"):  # Ignore firmware comments/boot messages
                    continue

                row = parse_row(line)
                if row is None:
                    skipped += 1
                    continue

                writer.writerow(row)
                f.flush()
                # Periodic fsync every 100 rows to prevent blocking I/O
                if rows % 100 == 0:
                    os.fsync(f.fileno())
                rows += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as e:
        print(f"\nSerial error: {e}")
    finally:
        try:
            if ser.is_open:
                ser.close()
        except Exception:
            pass
        print(f"Saved {rows} rows to {filename} ({skipped} malformed lines skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
