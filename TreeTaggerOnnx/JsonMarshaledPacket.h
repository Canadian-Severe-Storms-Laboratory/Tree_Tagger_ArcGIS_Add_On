#pragma once
#include "pch.h"
#include "json.hpp"
#include "Utils.h"
#include "Console.h"

using namespace Utils;

class JsonMarshaledPacket {
	using json = nlohmann::json;

public:

	template<typename T>
    static T unpack(const char* packet) {

        //converting to string instead of string view makes a copy the data
        try {
            json j = json::parse(std::string_view(packet));

            return T(j);
        }
        catch(std::exception ex) {

            throw std::runtime_error("Failed to parse packet");

            /*Console::consoleCallback("Failed to parse packet\n\n");
            printStackTrace();
            exit(1);*/
        }
    }

    template<typename T>
    static char* pack(T& t) {

	    const std::string str = t.toJson().dump();

        char* packet = new char[str.size() + 1];

        memcpy(packet, str.c_str(), str.size() + 1);

        return packet;
    }

    template<typename... Bases>
    static auto parsePacket(const char* packet) {

        return [&packet]() {

            struct _PacketType : public Bases... {

                _PacketType(nlohmann::json & j) : Bases(j)... {};
            };

            return unpack<_PacketType>(packet);

        }();
    }

};
