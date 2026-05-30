// C++ 测试 helper 用于验证 LiteIM 协议实现的编码和解析是否符合预期的协议契约。通过定义一系列合同测试用例，涵盖了不同类型的消息和字段组合，确保协议的正确性和兼容性。
#include "liteim/base/ErrorCode.hpp"
#include "liteim/base/Status.hpp"
#include "liteim/base/Types.hpp"
#include "liteim/protocol/MessageType.hpp"
#include "liteim/protocol/Packet.hpp"
#include "liteim/protocol/Tlv.hpp"
#include "liteim/protocol/TlvCodec.hpp"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

enum class FieldKind {
    String,
    Uint64,
};

struct Field {
    liteim::TlvType type;
    FieldKind kind;
    std::string text;
    std::uint64_t number{0};
};

struct ContractCase {
    std::string name;
    liteim::MessageType msg_type;
    std::uint64_t seq_id;
    std::vector<Field> fields;
};

Field stringField(liteim::TlvType type, std::string value) {
    return Field{type, FieldKind::String, std::move(value), 0};
}

Field uint64Field(liteim::TlvType type, std::uint64_t value) {
    return Field{type, FieldKind::Uint64, {}, value};
}

std::vector<ContractCase> contractCases() {
    return {
        {"login_request",
         liteim::MessageType::LoginRequest,
         1001,
         {
             stringField(liteim::TlvType::Username, "agent_bot"),
             stringField(liteim::TlvType::Password, "correct horse"),
         }},
        {"private_message_request",
         liteim::MessageType::PrivateMessageRequest,
         1002,
         {
             uint64Field(liteim::TlvType::ReceiverId, 2002),
             stringField(liteim::TlvType::ClientMessageId, "agent-msg-0001"),
             stringField(liteim::TlvType::MessageText, "hello bob 你好"),
         }},
        {"private_message_push",
         liteim::MessageType::PrivateMessagePush,
         0,
         {
             uint64Field(liteim::TlvType::MessageId, 5001),
             uint64Field(liteim::TlvType::ConversationType, 1),
             uint64Field(liteim::TlvType::ConversationId, 10012002),
             uint64Field(liteim::TlvType::SenderId, 1001),
             uint64Field(liteim::TlvType::ReceiverId, 2002),
             stringField(liteim::TlvType::MessageText, "push from alice"),
             stringField(liteim::TlvType::ClientMessageId, "alice-client-1"),
             uint64Field(liteim::TlvType::TimestampMs, 1700000001000ULL),
         }},
        {"offline_messages_ack_request",
         liteim::MessageType::OfflineMessagesAckRequest,
         1003,
         {
             uint64Field(liteim::TlvType::MessageId, 5001),
             uint64Field(liteim::TlvType::MessageId, 5002),
         }},
        {"delivery_ack_request",
         liteim::MessageType::DeliveryAckRequest,
         1004,
         {
             uint64Field(liteim::TlvType::MessageId, 5001),
         }},
        {"read_ack_request",
         liteim::MessageType::ReadAckRequest,
         1005,
         {
             uint64Field(liteim::TlvType::ConversationType, 1),
             uint64Field(liteim::TlvType::ConversationId, 10012002),
             uint64Field(liteim::TlvType::MessageId, 5002),
         }},
        {"history_response",
         liteim::MessageType::HistoryResponse,
         1006,
         {
             uint64Field(liteim::TlvType::MessageId, 5002),
             uint64Field(liteim::TlvType::ConversationType, 1),
             uint64Field(liteim::TlvType::ConversationId, 10012002),
             uint64Field(liteim::TlvType::SenderId, 2002),
             uint64Field(liteim::TlvType::ReceiverId, 1001),
             stringField(liteim::TlvType::MessageText, "newer history"),
             stringField(liteim::TlvType::ClientMessageId, "bob-client-2"),
             uint64Field(liteim::TlvType::TimestampMs, 1700000002000ULL),
             uint64Field(liteim::TlvType::MessageId, 5001),
             uint64Field(liteim::TlvType::ConversationType, 1),
             uint64Field(liteim::TlvType::ConversationId, 10012002),
             uint64Field(liteim::TlvType::SenderId, 1001),
             uint64Field(liteim::TlvType::ReceiverId, 2002),
             stringField(liteim::TlvType::MessageText, "older history"),
             stringField(liteim::TlvType::ClientMessageId, "alice-client-1"),
             uint64Field(liteim::TlvType::TimestampMs, 1700000001000ULL),
         }},
        {"error_response",
         liteim::MessageType::ErrorResponse,
         1007,
         {
             uint64Field(liteim::TlvType::ErrorCode, 5),
             stringField(liteim::TlvType::ErrorMessage, "invalid packet magic"),
         }},
    };
}

const ContractCase* findCase(const std::string& name, const std::vector<ContractCase>& cases) {
    for (const auto& contract_case : cases) {
        if (contract_case.name == name) {
            return &contract_case;
        }
    }
    return nullptr;
}

void failStatus(const char* operation, const liteim::Status& status) {
    std::cerr << operation << " failed: " << liteim::toString(status.code()) << ": "
              << status.message() << '\n';
    std::exit(2);
}

void failUsage(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(64);
}

std::string bytesToHex(const liteim::Byte* data, std::size_t size) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(data[index]);
    }
    return output.str();
}

std::string bytesToHex(const liteim::Bytes& bytes) {
    return bytesToHex(bytes.data(), bytes.size());
}

int hexDigit(char value) {
    const auto as_unsigned = static_cast<unsigned char>(value);
    if (std::isdigit(as_unsigned)) {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

bool parseHex(const std::string& hex, liteim::Bytes& output, std::string& error) {
    output.clear();
    if (hex.size() % 2 != 0) {
        error = "hex input length must be even";
        return false;
    }

    output.reserve(hex.size() / 2);
    for (std::size_t index = 0; index < hex.size(); index += 2) {
        const int high = hexDigit(hex[index]);
        const int low = hexDigit(hex[index + 1]);
        if (high < 0 || low < 0) {
            error = "hex input contains a non-hex character";
            return false;
        }
        output.push_back(static_cast<liteim::Byte>((high << 4U) | low));
    }
    return true;
}

void appendOrDie(liteim::Packet& packet, const Field& field) {
    liteim::Status status;
    if (field.kind == FieldKind::String) {
        status = liteim::appendString(field.type, field.text, packet.body);
    } else {
        status = liteim::appendUint64(field.type, field.number, packet.body);
    }
    if (!status.isOk()) {
        failStatus("append field", status);
    }
}

liteim::Packet packetFor(const ContractCase& contract_case) {
    liteim::Packet packet;
    packet.header.msg_type = contract_case.msg_type;
    packet.header.seq_id = contract_case.seq_id;
    for (const auto& field : contract_case.fields) {
        appendOrDie(packet, field);
    }
    return packet;
}

int encodeCase(const std::string& name) {
    const auto cases = contractCases();
    const auto* contract_case = findCase(name, cases);
    if (contract_case == nullptr) {
        failUsage("unknown contract case: " + name);
    }

    liteim::Bytes encoded;
    const auto status = liteim::encodePacket(packetFor(*contract_case), encoded);
    if (!status.isOk()) {
        failStatus("encode packet", status);
    }

    std::cout << "{\"hex\":\"" << bytesToHex(encoded) << "\"}\n";
    return 0;
}

int parsePacket(const std::string& hex) {
    liteim::Bytes encoded;
    std::string error;
    if (!parseHex(hex, encoded, error)) {
        std::cerr << "parse hex failed: InvalidArgument: " << error << '\n';
        return 2;
    }

    liteim::PacketHeader header;
    const auto header_status = liteim::parseHeader(encoded.data(), encoded.size(), header);
    if (!header_status.isOk()) {
        failStatus("parse header", header_status);
    }

    const auto expected_size =
        liteim::kPacketHeaderSize + static_cast<std::size_t>(header.body_len);
    if (encoded.size() != expected_size) {
        std::cerr << "parse packet failed: ParseError: packet size does not match body length\n";
        return 2;
    }

    const liteim::Byte* body_data = encoded.data() + liteim::kPacketHeaderSize;
    liteim::TlvMap fields;
    const auto tlv_status = liteim::parseTlvMap(body_data, header.body_len, fields);
    if (!tlv_status.isOk()) {
        failStatus("parse tlv map", tlv_status);
    }

    std::cout << "{\"msg_type\":" << static_cast<std::uint16_t>(header.msg_type)
              << ",\"msg_type_name\":\"" << liteim::toString(header.msg_type) << "\""
              << ",\"seq_id\":" << header.seq_id << ",\"body_len\":" << header.body_len
              << ",\"fields\":{";

    bool first_field = true;
    for (const auto& item : fields) {
        if (!first_field) {
            std::cout << ',';
        }
        first_field = false;

        std::cout << '\"' << static_cast<std::uint16_t>(item.first) << "\":[";
        bool first_value = true;
        for (const auto& value : item.second) {
            if (!first_value) {
                std::cout << ',';
            }
            first_value = false;
            std::cout << '\"' << bytesToHex(value) << '\"';
        }
        std::cout << ']';
    }
    std::cout << "}}\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        failUsage("usage: liteim_contract_helper <encode|parse> <case-name|packet-hex>");
    }

    const std::string command = argv[1];
    const std::string argument = argv[2];
    if (command == "encode") {
        return encodeCase(argument);
    }
    if (command == "parse") {
        return parsePacket(argument);
    }

    failUsage("unknown command: " + command);
}
