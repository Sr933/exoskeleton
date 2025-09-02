#pragma once

//#define ARDUINO
#ifdef ARDUINO
  #include <String.h>
  #include <WString.h>
  #include <stdint.h>
#else
  #include <string>
  #include <vector>
  #include <iostream>
#include "ArduinoString.h"
  using namespace std;
	using namespace EmotiBit;
#endif


class EmotiBitPacket {
public:

	// Platform independent fields
	// ToDo: Needs some refactoring for direct binary transmision
	class Header {
	public:
		uint32_t timestamp;
		uint16_t packetNumber;
		uint16_t dataLength;
#ifdef ARDUINO
		String typeTag;
#else
		std::string typeTag;
#endif
		uint8_t protocolVersion;
		uint8_t dataReliability;
		
		struct Structure {
			uint32_t timestamp;
			uint16_t packetNumber;
			uint16_t dataLength;
			char typeTag[2];
			uint8_t protocolVersion;
			uint8_t dataReliability;
		};
	};

	static const uint8_t headerLength = 6;
	static const uint8_t headerByteLength = 12;
	static const uint8_t maxHeaderCharLength = 35; // 13+(1)+5+(1)+3+(1)+2+(1)+3+(1)+3+(1)

	static const char PAYLOAD_DELIMITER = ',';
	static const char PAYLOAD_TRUNCATED = (char) 25;	// EM character, End of Medium
	
	//Header header;
#ifdef ARDUINO
	static Header createHeader(const String &typeTag, uint32_t timestamp, uint16_t packetNumber, uint16_t dataLength = 0, uint8_t protocolVersion = 1, uint8_t dataReliability = 100);
#else
	static Header createHeader(const string &typeTag, uint32_t timestamp, uint16_t packetNumber, uint16_t dataLength = 0, uint8_t protocolVersion = 1, uint8_t dataReliability = 100);
#endif

#ifdef ARDUINO
	static String headerToString(const Header &header);
#else
	static string headerToString(const Header &header);
#endif


	static const int8_t FAIL = -1;
	static const int16_t NO_PACKET_DATA = -2;
	static const int16_t MALFORMED_HEADER = -3;


#ifdef ARDUINO
	// Arduino specific fields
#else
	// Non-arduino STL fields
	//vector<string> data;
#endif

	// Platform independent functions
	//EmotiBitPacket();
	//bool parsePacket(string packet);

#ifdef ARDUINO
	// Arduino specific functions
	//static String getTypeTag(const Header &h);
#else
	// Non-arduino STL functions
	//string getTypeTag();
	//static string getTypeTag(const Header &h);
	//Header getHeader();

#endif

	enum class PacketType {
		// EmotiBit Sensor Data
		EDA,	// Electrodermal Activity
		EDL,	// Electrodermal Level
		EDR,	// Electrodermal Response
		PPG_INFRARED,	// PPG Infrared
		PPG_RED,	// PPG Red
		PPG_GREEN,	// PPG Green
		SPO2,	// SpO2
		TEMPERATURE_0,	// Temperature 0
		TEMPERATURE_1, // Temperature measured on EmotiBit bottom, closest to center
		THERMOPILE,	// Thermistor
		HUMIDITY_0,	// Humidity 0
		ACCELEROMETER_X,	// Accelerometer X-axis
		ACCELEROMETER_Y,	// Accelerometer Y-axis
		ACCELEROMETER_Z,	// Accelerometer Z-axis
		GYROSCOPE_X,	// Gyroscope X-axis
		GYROSCOPE_Y,	// Gyroscope Y-axis
		GYROSCOPE_Z,	// Gyroscope Z-axis
		MAGNETOMETER_X,	// Magnetometer X-axis
		MAGNETOMETER_Y,	// Magnetometer Y-axis
		MAGNETOMETER_Z,	// Magnetometer Z-axis
		// Additional EmotiBit Info
		BATTERY_VOLTAGE,	// Battery Voltage
		BATTERY_PERCENT,	// Battery percent(%)
		BUTTON_PRESS_SHORT, // Button Press Short
		BUTTON_PRESS_LONG, // Button Press Long
		DATA_CLIPPING,	// Data clipping, data = typeTags of data
		DATA_OVERFLOW,	// Data overflow, data = typeTags of data
		SD_CARD_PERCENT,	// SD card % full
		RESET,	// 
		// Computer -> EmotiBit Comms
		GPS_LATLNG,	// GPS Latitude & Longitude, data = latitude, longitude, accuracy
		GPS_SPEED,	// GPS Speed, data = speed, accuracy
		GPS_BEARING,	// GPS bearing, data = bearing, accuracy
		GPS_ALTITUDE,	// GPS altitude, data = altitude, verticalAccuracy
		TIMESTAMP_LOCAL,	// Local computer timestamp
		TIMESTAMP_UTC,	// UTC timestamp
		TIMESTAMP_CROSS_TIME, //Used for comparison between different clocks
		LSL_MARKER,
		RECORD_BEGIN,	// Record begin
		RECORD_END,	// Record end
		USER_NOTE,	// User note, data = char array
		MODE_HIBERNATE,	// Mode hibernate
		HELLO_EMOTIBIT,	// Hello Emotibit (used to establish communications)
		// General Comms
		ACK,
		NACK,
		REQUEST_DATA,	// Request data, data = list of requested data types
		PING,	// Ping
		PONG,	// Pong
		length	// number of packet types
	};

	//static const char* typeTags[]; // Iterable reference to all typeTags

	class TypeTag 
	{
	public:
		static const char* EDA;
		static const char* EDL;
		static const char* EDR;
		static const char* PPG_INFRARED;
		static const char* PPG_RED;
		static const char* PPG_GREEN;
		static const char* SPO2;
		static const char* TEMPERATURE_0;
		static const char* TEMPERATURE_1;
		static const char* THERMOPILE;
		static const char* HUMIDITY_0;
		static const char* ACCELEROMETER_X;
		static const char* ACCELEROMETER_Y;
		static const char* ACCELEROMETER_Z;
		static const char* GYROSCOPE_X;
		static const char* GYROSCOPE_Y;
		static const char* GYROSCOPE_Z;
		static const char* MAGNETOMETER_X;
		static const char* MAGNETOMETER_Y;
		static const char* MAGNETOMETER_Z;
		static const char* BATTERY_VOLTAGE;
		static const char* BATTERY_PERCENT;
		static const char* BUTTON_PRESS_SHORT;
		static const char* BUTTON_PRESS_LONG;
		static const char* DATA_CLIPPING;
		static const char* DATA_OVERFLOW;
		static const char* SD_CARD_PERCENT;
		static const char* RESET;
		static const char* GPS_LATLNG;
		static const char* GPS_SPEED;
		static const char* GPS_BEARING;
		static const char* GPS_ALTITUDE;
		static const char* TIMESTAMP_EMOTIBIT;
		static const char* TIMESTAMP_LOCAL;
		static const char* TIMESTAMP_UTC;
		static const char* TIMESTAMP_CROSS_TIME;
		static const char* LSL_MARKER;
		static const char* EMOTIBIT_MODE;
		static const char* EMOTIBIT_INFO;
		static const char* RECORD_BEGIN;
		static const char* RECORD_END;
		static const char* USER_NOTE;
		static const char* MODE_NORMAL_POWER;
		static const char* MODE_LOW_POWER;
		static const char* MODE_MAX_LOW_POWER;
		static const char* MODE_WIRELESS_OFF;
		static const char* MODE_HIBERNATE;
		static const char* HELLO_EMOTIBIT;
		static const char* HELLO_HOST;
		//static const char* HELLO_COMPUTER;
		static const char* EMOTIBIT_CONNECT;
		static const char* EMOTIBIT_DISCONNECT;
		static const char* SERIAL_DATA_ON;
		static const char* SERIAL_DATA_OFF;
		static const char* WIFI_ADD;
		static const char* WIFI_DELETE;
		static const char* LIST;
		//static const char* KEEP_ALIVE;
		static const char* ACK;
		static const char* NACK;
		static const char* REQUEST_DATA;
		static const char* PING;
		static const char* PONG;
		static const char* EMOTIBIT_DEBUG;
		static const char* HEART_RATE;
		static const char* INTER_BEAT_INTERVAL;
		static const char* SKIN_CONDUCTANCE_RESPONSE_AMPLITUDE;
		static const char* SKIN_CONDUCTANCE_RESPONSE_FREQ;
		static const char* SKIN_CONDUCTANCE_RESPONSE_RISE_TIME;
//        static vector<string> APERIODIC;
	};
    
  class TypeTagGroups
	{
  public:
	  static uint8_t NUM_APERIODIC;
	  static uint8_t NUM_USER_MESSAGES;
	  static uint8_t NUM_COMPOSITE_PAYLOAD;
      static const char* const APERIODIC[];
      static const char* const USER_MESSAGES[];
	  static const char* const COMPOSITE_PAYLOAD[];  //!< Array to hold typetags which contain multiple types of information in payload
  };

	class PayloadLabel
	{
	public:
		static const char* CONTROL_PORT;
		static const char* DATA_PORT;
		static const char* DEVICE_ID;
		static const char* RECORDING_STATUS;
		static const char* POWER_STATUS;
		static const char* LSL_MARKER_RX_TIMESTAMP;
		static const char* LSL_MARKER_SRC_TIMESTAMP;
		static const char* LSL_LOCAL_CLOCK_TIMESTAMP;
		static const char* LSL_MARKER_DATA;
	};
	
  static const char PACKET_DELIMITER_CSV;
	static const uint16_t MAX_TO_EMOTIBIT_PACKET_LEN = 255;
#ifdef ARDUINO
	static const String TIMESTAMP_STRING_FORMAT;
#else
	static const string TIMESTAMP_STRING_FORMAT;
#endif

	EmotiBitPacket();

	

	static int16_t getHeader(const String & packet, Header &packetHeader); // Returns position of data start character, FAIL if malformed
	
	/// Extracts a single packet element from the passed packet string
	/// @param packet is the string parsed for the next element
	/// @param element is filled with the extracted element
	/// @return startChar of the next element or -1 if no following packet exists
	static int16_t getPacketElement(const String &packet, String &element, uint16_t startChar = 0);

	/// Extracts a value packet element following the key packet element
	/// @param packet is the string parsed for packet elements
	/// @param key to search for in the packet
	/// @param value is filled with the extracted element
	/// @return startChar of the value element or -1 if no key/value exists
	static int16_t getPacketKeyedValue(const String &packet, const String &key, String &value, int16_t startChar = 0);

#ifndef ARDUINO
	static bool getHeader(const vector<string>& packet, Header &packetHeader); // Returns false if the packet is malformed
	static int16_t getPacketElement(const string &packet, string &element, uint16_t startChar = 0);
	static int16_t getPacketKeyedValue(const string &packet, const string &key, string &value, uint16_t startChar = 0);
	static int16_t getPacketKeyedValue(const vector<string> &splitPacket, const string &key, string &value, 
		uint16_t startIndex = EmotiBitPacket::headerLength);
#endif


#ifdef ARDUINO
	static String createPacket(const String &typeTag, const uint16_t &packetNumber, const String &data, const uint16_t &dataLength, const uint8_t &protocolVersion = 1, const uint8_t &dataReliability = 100);
#else
	static string createPacket(const string &typeTag, const uint16_t &packetNumber, const string &data, const uint16_t &dataLength, const uint8_t &protocolVersion = 1, const uint8_t &dataReliability = 100);
	static string createPacket(const string &typeTag, const uint16_t &packetNumber, const vector<string> &data, const uint8_t &protocolVersion = 1, const uint8_t &dataReliability = 100);
	
#endif
private:
	
};
