#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


HANDLE_DEFAULT = "64"


CODE_NAMES = {
    0x0000: "FUN_SENDER_DEVICE_INFO",
    0x0001: "FUN_SENDER_PAIR",
    0x0002: "FUN_SENDER_DPS",
    0x0003: "FUN_SENDER_DEVICE_STATUS",
    0x0005: "FUN_SENDER_UNBIND",
    0x0006: "FUN_SENDER_DEVICE_RESET",
    0x000C: "FUN_SENDER_OTA_START",
    0x000D: "FUN_SENDER_OTA_FILE",
    0x000E: "FUN_SENDER_OTA_OFFSET",
    0x000F: "FUN_SENDER_OTA_UPGRADE",
    0x0010: "FUN_SENDER_OTA_OVER",
    0x0027: "FUN_SENDER_DPS_V4",
    0x8001: "FUN_RECEIVE_DP",
    0x8003: "FUN_RECEIVE_TIME_DP",
    0x8004: "FUN_RECEIVE_SIGN_DP",
    0x8005: "FUN_RECEIVE_SIGN_TIME_DP",
    0x8006: "FUN_RECEIVE_DP_V4",
    0x8007: "FUN_RECEIVE_TIME_DP_V4",
    0x8011: "FUN_RECEIVE_TIME1_REQ",
    0x8012: "FUN_RECEIVE_TIME2_REQ",
}


@dataclass
class Fragment:
    line_no: int
    direction: str
    payload: bytes


@dataclass
class Message:
    line_no: int
    direction: str
    payload: bytes


def unpack_int(data: bytes, start_pos: int) -> tuple[int, int]:
    result = 0
    offset = 0
    while offset < 5:
        pos = start_pos + offset
        if pos >= len(data):
            raise ValueError("short varint")
        curr = data[pos]
        result |= (curr & 0x7F) << (offset * 7)
        offset += 1
        if (curr & 0x80) == 0:
            break
    return result, start_pos + offset


def aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def decode_datapoints(data: bytes, start_pos: int = 0) -> list[tuple[int, int, int, object]]:
    pos = start_pos
    items: list[tuple[int, int, int, object]] = []
    while len(data) - pos >= 4:
        dp_id = data[pos]
        dp_type = data[pos + 1]
        dp_len = data[pos + 2]
        pos += 3
        raw = data[pos : pos + dp_len]
        pos += dp_len

        if dp_type == 0:
            value: object = raw.hex()
        elif dp_type == 1:
            value = int.from_bytes(raw, "big") != 0
        elif dp_type in (2, 4):
            value = int.from_bytes(raw, "big", signed=True)
        elif dp_type == 3:
            try:
                value = raw.decode()
            except Exception:
                value = raw.hex()
        elif dp_type == 5:
            value = raw.hex()
        else:
            value = raw.hex()
        items.append((dp_id, dp_type, dp_len, value))
    return items


def collect_att_fragments(text: str, handle: str) -> list[Fragment]:
    fragments: list[Fragment] = []
    current_direction: str | None = None
    capture_payload = False

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if f"ACL Data TX: Handle {handle}" in raw_line:
            current_direction = "TX"
            capture_payload = False
            continue
        if f"ACL Data RX: Handle {handle}" in raw_line:
            current_direction = "RX"
            capture_payload = False
            continue
        if current_direction and re.search(r"ATT: (Write Command|Handle Value Notification)", raw_line):
            capture_payload = True
            continue
        match = re.search(r"\s+Data\[\d+\]: ([0-9a-f]+)", raw_line)
        if match and current_direction and capture_payload:
            fragments.append(
                Fragment(
                    line_no=line_no,
                    direction=current_direction,
                    payload=bytes.fromhex(match.group(1)),
                )
            )
            capture_payload = False

    return fragments


def assemble_messages(fragments: list[Fragment]) -> list[Message]:
    current: dict[str, dict[str, object] | None] = {"TX": None, "RX": None}
    messages: list[Message] = []

    for fragment in fragments:
        try:
            packet_num, pos = unpack_int(fragment.payload, 0)
        except ValueError:
            continue

        state = current[fragment.direction]

        if packet_num == 0:
            if state and state["buf"]:
                messages.append(
                    Message(
                        line_no=state["line_no"],  # type: ignore[arg-type]
                        direction=state["direction"],  # type: ignore[arg-type]
                        payload=bytes(state["buf"]),  # type: ignore[arg-type]
                    )
                )

            try:
                expected_len, pos = unpack_int(fragment.payload, pos)
            except ValueError:
                current[fragment.direction] = None
                continue

            if pos >= len(fragment.payload):
                current[fragment.direction] = None
                continue

            current[fragment.direction] = {
                "line_no": fragment.line_no,
                "direction": fragment.direction,
                "expected_len": expected_len,
                "buf": bytearray(fragment.payload[pos + 1 :]),
            }
        else:
            if not state:
                continue
            state["buf"].extend(fragment.payload[pos:])  # type: ignore[union-attr]

        state = current[fragment.direction]
        if state and len(state["buf"]) >= state["expected_len"]:  # type: ignore[arg-type]
            messages.append(
                Message(
                    line_no=state["line_no"],  # type: ignore[arg-type]
                    direction=state["direction"],  # type: ignore[arg-type]
                    payload=bytes(state["buf"]),  # type: ignore[arg-type]
                )
            )
            current[fragment.direction] = None

    for direction in ("TX", "RX"):
        state = current[direction]
        if state and state["buf"]:
            messages.append(
                Message(
                    line_no=state["line_no"],  # type: ignore[arg-type]
                    direction=state["direction"],  # type: ignore[arg-type]
                    payload=bytes(state["buf"]),  # type: ignore[arg-type]
                )
            )

    messages.sort(key=lambda msg: msg.line_no)
    return messages


def decode_messages(messages: list[Message], local_key: bytes) -> list[str]:
    tuya_key = local_key[:6]
    login_key = hashlib.md5(tuya_key).digest()
    session_key: bytes | None = None
    auth_key: bytes | None = None
    session_no = 0
    lines: list[str] = []

    for idx, msg in enumerate(messages):
        enc = msg.payload
        if len(enc) < 17:
            lines.append(f"[{idx:03d}] {msg.direction} short_message len={len(enc)} payload={enc.hex()}")
            continue

        security_flag = enc[0]
        iv = enc[1:17]
        ciphertext = enc[17:]

        if security_flag == 4:
            key = login_key
        elif security_flag == 5:
            key = session_key
        elif security_flag == 1:
            key = auth_key
        else:
            key = None

        if key is None:
            lines.append(
                f"[{idx:03d}] {msg.direction} sec={security_flag} len={len(enc)} "
                f"(no key yet) payload={enc.hex()}"
            )
            continue

        try:
            raw = aes_cbc_decrypt(key, iv, ciphertext)
        except Exception as exc:
            lines.append(f"[{idx:03d}] {msg.direction} sec={security_flag} decrypt_error={exc} payload={enc.hex()}")
            continue

        if len(raw) < 12:
            lines.append(f"[{idx:03d}] {msg.direction} sec={security_flag} raw_too_short len={len(raw)} payload={enc.hex()}")
            continue

        seq_num, response_to, code_val, length = struct.unpack(">IIHH", raw[:12])
        if len(raw) < 12 + length:
            lines.append(
                f"[{idx:03d}] {msg.direction} sec={security_flag} raw_short_for_len "
                f"seq={seq_num} code=0x{code_val:04x} need={length} have={len(raw) - 12}"
            )
            continue

        data = raw[12 : 12 + length]
        code_name = CODE_NAMES.get(code_val, f"0x{code_val:04x}")

        if msg.direction == "TX" and security_flag == 4 and code_val == 0x0000 and response_to == 0:
            session_no += 1
            session_key = None
            auth_key = None
            lines.append(f"=== Session {session_no} start (line {msg.line_no}) ===")

        lines.append(
            f"[{idx:03d}] {msg.direction} sec={security_flag} seq={seq_num} "
            f"resp_to={response_to} code={code_name} len={length} data={data.hex()}"
        )

        if security_flag == 4 and code_val == 0x0000 and response_to != 0 and len(data) >= 46:
            srand = data[6:12]
            session_key = hashlib.md5(tuya_key + srand).digest()
            auth_key = data[14:46]
            lines.append(f"      derived session_key={session_key.hex()} auth_key={auth_key.hex()}")

        if code_val in (0x8001, 0x8004, 0x8006):
            start_pos = 3 if code_val == 0x8004 else 0
            dp_items = decode_datapoints(data, start_pos)
            if dp_items:
                pretty = ", ".join(
                    f"dp{dp_id}:t{dp_type}:len{dp_len}={value}"
                    for dp_id, dp_type, dp_len, value in dp_items
                )
                lines.append(f"      datapoints: {pretty}")
        elif code_val == 0x0002 and len(data) == 1:
            if data == b"\x01":
                lines.append("      empty sender DPS ack/keepalive")
            else:
                lines.append(f"      sender DPS payload: {data.hex()}")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode Tuya BLE packets from btmon/btsnoop text output.")
    parser.add_argument("input", type=Path, help="Path to btmon text output")
    parser.add_argument(
        "--local-key",
        required=True,
        help="Tuya local key for the device (the full key, not the derived login key)",
    )
    parser.add_argument(
        "--handle",
        default=HANDLE_DEFAULT,
        help=f"Bluetooth handle to decode (default: {HANDLE_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the decoded transcript to this path instead of stdout",
    )
    args = parser.parse_args()

    text = args.input.read_text(errors="ignore")
    fragments = collect_att_fragments(text, args.handle)
    messages = assemble_messages(fragments)
    lines = [
        f"Local key: {args.local_key}",
        f"Login key md5: {hashlib.md5(args.local_key.encode()[:6]).hexdigest()}",
        f"Captured ATT payloads: {len(fragments)}",
        f"Reassembled messages: {len(messages)}",
        "",
    ]
    lines.extend(decode_messages(messages, args.local_key.encode()))
    output = "\n".join(lines) + "\n"

    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
