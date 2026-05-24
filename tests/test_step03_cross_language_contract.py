from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from bot_client.frame_decoder import FrameDecoder
from bot_client.liteim_protocol import (
    MessageType,
    Packet,
    PacketHeader,
    ProtocolError,
    TlvMap,
    TlvType,
    append_string,
    append_uint64,
    encode_packet,
    parse_tlv_map,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LITEIM_ROOT = WORKSPACE_ROOT / "LiteIM"
HELPER_SOURCE = PROJECT_ROOT / "tests" / "cpp" / "liteim_contract_helper.cpp"


@dataclass(frozen=True)
class ContractField:
    tlv_type: TlvType
    kind: Literal["string", "uint64"]
    value: str | int


@dataclass(frozen=True)
class ContractCase:
    name: str
    msg_type: MessageType
    seq_id: int
    fields: tuple[ContractField, ...]


CONTRACT_CASES = (
    ContractCase(
        name="login_request",
        msg_type=MessageType.LoginRequest,
        seq_id=1001,
        fields=(
            ContractField(TlvType.Username, "string", "agent_bot"),
            ContractField(TlvType.Password, "string", "correct horse"),
        ),
    ),
    ContractCase(
        name="private_message_request",
        msg_type=MessageType.PrivateMessageRequest,
        seq_id=1002,
        fields=(
            ContractField(TlvType.ReceiverId, "uint64", 2002),
            ContractField(TlvType.ClientMessageId, "string", "agent-msg-0001"),
            ContractField(TlvType.MessageText, "string", "hello bob 你好"),
        ),
    ),
    ContractCase(
        name="private_message_push",
        msg_type=MessageType.PrivateMessagePush,
        seq_id=0,
        fields=(
            ContractField(TlvType.MessageId, "uint64", 5001),
            ContractField(TlvType.ConversationType, "uint64", 1),
            ContractField(TlvType.ConversationId, "uint64", 10012002),
            ContractField(TlvType.SenderId, "uint64", 1001),
            ContractField(TlvType.ReceiverId, "uint64", 2002),
            ContractField(TlvType.MessageText, "string", "push from alice"),
            ContractField(TlvType.ClientMessageId, "string", "alice-client-1"),
            ContractField(TlvType.TimestampMs, "uint64", 1_700_000_001_000),
        ),
    ),
    ContractCase(
        name="offline_messages_ack_request",
        msg_type=MessageType.OfflineMessagesAckRequest,
        seq_id=1003,
        fields=(
            ContractField(TlvType.MessageId, "uint64", 5001),
            ContractField(TlvType.MessageId, "uint64", 5002),
        ),
    ),
    ContractCase(
        name="delivery_ack_request",
        msg_type=MessageType.DeliveryAckRequest,
        seq_id=1004,
        fields=(ContractField(TlvType.MessageId, "uint64", 5001),),
    ),
    ContractCase(
        name="read_ack_request",
        msg_type=MessageType.ReadAckRequest,
        seq_id=1005,
        fields=(
            ContractField(TlvType.ConversationType, "uint64", 1),
            ContractField(TlvType.ConversationId, "uint64", 10012002),
            ContractField(TlvType.MessageId, "uint64", 5002),
        ),
    ),
    ContractCase(
        name="history_response",
        msg_type=MessageType.HistoryResponse,
        seq_id=1006,
        fields=(
            ContractField(TlvType.MessageId, "uint64", 5002),
            ContractField(TlvType.ConversationType, "uint64", 1),
            ContractField(TlvType.ConversationId, "uint64", 10012002),
            ContractField(TlvType.SenderId, "uint64", 2002),
            ContractField(TlvType.ReceiverId, "uint64", 1001),
            ContractField(TlvType.MessageText, "string", "newer history"),
            ContractField(TlvType.ClientMessageId, "string", "bob-client-2"),
            ContractField(TlvType.TimestampMs, "uint64", 1_700_000_002_000),
            ContractField(TlvType.MessageId, "uint64", 5001),
            ContractField(TlvType.ConversationType, "uint64", 1),
            ContractField(TlvType.ConversationId, "uint64", 10012002),
            ContractField(TlvType.SenderId, "uint64", 1001),
            ContractField(TlvType.ReceiverId, "uint64", 2002),
            ContractField(TlvType.MessageText, "string", "older history"),
            ContractField(TlvType.ClientMessageId, "string", "alice-client-1"),
            ContractField(TlvType.TimestampMs, "uint64", 1_700_000_001_000),
        ),
    ),
    ContractCase(
        name="error_response",
        msg_type=MessageType.ErrorResponse,
        seq_id=1007,
        fields=(
            ContractField(TlvType.ErrorCode, "uint64", 5),
            ContractField(TlvType.ErrorMessage, "string", "invalid packet magic"),
        ),
    ),
)


def _field_value_hex(field: ContractField) -> str:
    if field.kind == "string":
        assert isinstance(field.value, str)
        return field.value.encode("utf-8").hex()
    assert isinstance(field.value, int)
    return field.value.to_bytes(8, byteorder="big").hex()


def _expected_fields(fields: tuple[ContractField, ...]) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    for field in fields:
        expected.setdefault(str(int(field.tlv_type)), []).append(_field_value_hex(field))
    return expected


def _body_for(fields: tuple[ContractField, ...]) -> bytes:
    body = bytearray()
    for field in fields:
        if field.kind == "string":
            assert isinstance(field.value, str)
            append_string(field.tlv_type, field.value, body)
        else:
            assert isinstance(field.value, int)
            append_uint64(field.tlv_type, field.value, body)
    return bytes(body)


def _fields_from_python_tlv_map(tlv_map: TlvMap) -> dict[str, list[str]]:
    return {
        str(raw_type): [value.hex() for value in values]
        for raw_type, values in tlv_map.items()
    }


@pytest.fixture(scope="session")
def liteim_contract_helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HELPER_SOURCE.exists():
        pytest.fail(f"missing LiteIM contract helper source: {HELPER_SOURCE}")
    if not LITEIM_ROOT.exists():
        pytest.skip(f"LiteIM sibling repository is not available: {LITEIM_ROOT}")

    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("g++ is required for Step 03 LiteIM C++ contract tests")

    output_dir = tmp_path_factory.mktemp("liteim_contract")
    binary = output_dir / "liteim_contract_helper"
    command = [
        cxx,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-I",
        str(LITEIM_ROOT / "include"),
        str(HELPER_SOURCE),
        str(LITEIM_ROOT / "src" / "base" / "ErrorCode.cpp"),
        str(LITEIM_ROOT / "src" / "base" / "Status.cpp"),
        str(LITEIM_ROOT / "src" / "protocol" / "Packet.cpp"),
        str(LITEIM_ROOT / "src" / "protocol" / "TlvCodec.cpp"),
        str(LITEIM_ROOT / "src" / "protocol" / "MessageType.cpp"),
        str(LITEIM_ROOT / "src" / "protocol" / "Tlv.cpp"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return binary


def _run_helper_json(helper: Path, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [str(helper), *args],
        check=True,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def _run_helper_error(helper: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(helper), *args],
        check=False,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    return result


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[case.name for case in CONTRACT_CASES])
def test_python_encoded_packets_are_parsed_by_liteim_cpp(
    liteim_contract_helper: Path, case: ContractCase
) -> None:
    body = _body_for(case.fields)
    encoded = encode_packet(
        Packet(header=PacketHeader(msg_type=case.msg_type, seq_id=case.seq_id), body=body)
    )

    parsed = _run_helper_json(liteim_contract_helper, "parse", encoded.hex())

    assert parsed["msg_type"] == int(case.msg_type)
    assert parsed["seq_id"] == case.seq_id
    assert parsed["body_len"] == len(body)
    assert parsed["fields"] == _expected_fields(case.fields)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[case.name for case in CONTRACT_CASES])
def test_liteim_cpp_encoded_packets_are_parsed_by_python(
    liteim_contract_helper: Path, case: ContractCase
) -> None:
    helper_output = _run_helper_json(liteim_contract_helper, "encode", case.name)
    encoded = bytes.fromhex(str(helper_output["hex"]))

    packets = FrameDecoder().feed(encoded)

    assert len(packets) == 1
    assert packets[0].header.msg_type == case.msg_type
    assert packets[0].header.seq_id == case.seq_id
    assert packets[0].header.body_len == len(packets[0].body)
    assert _fields_from_python_tlv_map(parse_tlv_map(packets[0].body)) == _expected_fields(
        case.fields
    )


def test_invalid_packets_are_rejected_by_both_python_and_liteim_cpp(
    liteim_contract_helper: Path,
) -> None:
    valid = encode_packet(
        Packet(
            header=PacketHeader(msg_type=MessageType.LoginRequest, seq_id=2001),
            body=_body_for(CONTRACT_CASES[0].fields),
        )
    )
    invalid_magic = b"\x00" + valid[1:]

    with pytest.raises(ProtocolError):
        FrameDecoder().feed(invalid_magic)
    cpp_invalid_magic = _run_helper_error(liteim_contract_helper, "parse", invalid_magic.hex())
    assert "ParseError" in cpp_invalid_magic.stderr

    bad_tlv_body = bytearray(append_string(TlvType.Username, "a"))
    bad_tlv_body[5] = 2
    invalid_tlv_packet = encode_packet(
        Packet(
            header=PacketHeader(msg_type=MessageType.LoginRequest, seq_id=2002),
            body=bytes(bad_tlv_body),
        )
    )

    with pytest.raises(ProtocolError):
        parse_tlv_map(bytes(bad_tlv_body))
    cpp_invalid_tlv = _run_helper_error(liteim_contract_helper, "parse", invalid_tlv_packet.hex())
    assert "ParseError" in cpp_invalid_tlv.stderr
