from __future__ import annotations

from bot_client.liteim_protocol import (
    PACKET_HEADER_SIZE,
    BytesLike,
    Packet,
    ProtocolError,
    ProtocolErrorCode,
    parse_header,
)


class FrameDecoder:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._error = False

    @property
    def has_error(self) -> bool:
        return self._error

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._error = False

    def feed(self, data: BytesLike | None) -> list[Packet]:
        if self._error:
            raise ProtocolError(ProtocolErrorCode.ParseError, "frame decoder is in error state")
        if data is None:
            raise ProtocolError(ProtocolErrorCode.InvalidArgument, "frame decoder input is null")

        self._buffer.extend(bytes(data))
        packets: list[Packet] = []
        read_index = 0

        while len(self._buffer) - read_index >= PACKET_HEADER_SIZE:
            try:
                header = parse_header(
                    self._buffer[read_index : read_index + PACKET_HEADER_SIZE]
                )
            except ProtocolError:
                del self._buffer[:read_index]
                self._error = True
                raise

            frame_size = PACKET_HEADER_SIZE + header.body_len
            if len(self._buffer) - read_index < frame_size:
                break

            body_start = read_index + PACKET_HEADER_SIZE
            body_end = read_index + frame_size
            packets.append(Packet(header=header, body=bytes(self._buffer[body_start:body_end])))
            read_index += frame_size

        if read_index:
            del self._buffer[:read_index]

        return packets
