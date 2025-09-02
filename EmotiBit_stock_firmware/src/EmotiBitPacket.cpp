#include "EmotiBitPacket.h"

#ifdef ARDUINO
	#include <Arduino.h>
#else
	// ToDo: Remove OF dependency for ofToString()
	#include "ofMain.h"
#endif // !ARDUINO


// EmotiBit Data TagTypes
const char* EmotiBitPacket::TypeTag::EDA = "EA\0";
const char* EmotiBitPacket::TypeTag::EDL = "EL\0";
const char* EmotiBitPacket::TypeTag::EDR = "ER\0";
const char* EmotiBitPacket::TypeTag::PPG_INFRARED = "PI\0";
const char* EmotiBitPacket::TypeTag::PPG_RED = "PR\0";
const char* EmotiBitPacket::TypeTag::PPG_GREEN = "PG\0";
const char* EmotiBitPacket::TypeTag::SPO2 = "O2\0";
const char* EmotiBitPacket::TypeTag::TEMPERATURE_0 = "T0\0";
const char* EmotiBitPacket::TypeTag::TEMPERATURE_1 = "T1\0";
const char* EmotiBitPacket::TypeTag::THERMOPILE = "TH\0";
const char* EmotiBitPacket::TypeTag::HUMIDITY_0 = "H0\0";
const char* EmotiBitPacket::TypeTag::ACCELEROMETER_X = "AX\0";
const char* EmotiBitPacket::TypeTag::ACCELEROMETER_Y = "AY\0";
const char* EmotiBitPacket::TypeTag::ACCELEROMETER_Z = "AZ\0";
const char* EmotiBitPacket::TypeTag::GYROSCOPE_X = "GX\0";
const char* EmotiBitPacket::TypeTag::GYROSCOPE_Y = "GY\0";
const char* EmotiBitPacket::TypeTag::GYROSCOPE_Z = "GZ\0";
const char* EmotiBitPacket::TypeTag::MAGNETOMETER_X = "MX\0";
const char* EmotiBitPacket::TypeTag::MAGNETOMETER_Y = "MY\0";
const char* EmotiBitPacket::TypeTag::MAGNETOMETER_Z = "MZ\0";
const char* EmotiBitPacket::TypeTag::BATTERY_VOLTAGE = "BV\0";
const char* EmotiBitPacket::TypeTag::BATTERY_PERCENT = "B%\0";
const char* EmotiBitPacket::TypeTag::BUTTON_PRESS_SHORT = "BS\0";
const char* EmotiBitPacket::TypeTag::BUTTON_PRESS_LONG = "BL\0";
const char* EmotiBitPacket::TypeTag::DATA_CLIPPING = "DC\0";
const char* EmotiBitPacket::TypeTag::DATA_OVERFLOW = "DO\0";
const char* EmotiBitPacket::TypeTag::SD_CARD_PERCENT = "SD\0";
const char* EmotiBitPacket::TypeTag::RESET = "RS\0";
const char* EmotiBitPacket::TypeTag::EMOTIBIT_DEBUG = "DB\0";
const char* EmotiBitPacket::TypeTag::ACK = "AK\0";
const char* EmotiBitPacket::TypeTag::NACK = "NK\0";
const char* EmotiBitPacket::TypeTag::REQUEST_DATA = "RD\0";
const char* EmotiBitPacket::TypeTag::TIMESTAMP_EMOTIBIT = "TE\0";
const char* EmotiBitPacket::TypeTag::TIMESTAMP_LOCAL = "TL\0";
const char* EmotiBitPacket::TypeTag::TIMESTAMP_UTC = "TU\0";
const char* EmotiBitPacket::TypeTag::TIMESTAMP_CROSS_TIME = "TX\0";
const char* EmotiBitPacket::TypeTag::EMOTIBIT_MODE = "EM\0";
const char* EmotiBitPacket::TypeTag::EMOTIBIT_INFO = "EI\0";
const char* EmotiBitPacket::TypeTag::HEART_RATE = "HR\0";
const char* EmotiBitPacket::TypeTag::INTER_BEAT_INTERVAL = "BI\0";
const char* EmotiBitPacket::TypeTag::SKIN_CONDUCTANCE_RESPONSE_AMPLITUDE = "SA\0";
const char* EmotiBitPacket::TypeTag::SKIN_CONDUCTANCE_RESPONSE_FREQ = "SF\0";
const char* EmotiBitPacket::TypeTag::SKIN_CONDUCTANCE_RESPONSE_RISE_TIME = "SR\0";
// Computer data TypeTags (sent over reliable channel e.g. Control)
const char* EmotiBitPacket::TypeTag::GPS_LATLNG = "GL\0";
const char* EmotiBitPacket::TypeTag::GPS_SPEED = "GS\0";
const char* EmotiBitPacket::TypeTag::GPS_BEARING = "GB\0";
const char* EmotiBitPacket::TypeTag::GPS_ALTITUDE = "GA\0";
const char* EmotiBitPacket::TypeTag::USER_NOTE = "UN\0";
const char* EmotiBitPacket::TypeTag::LSL_MARKER = "LM\0";
// Control TypeTags
const char* EmotiBitPacket::TypeTag::RECORD_BEGIN = "RB\0";
const char* EmotiBitPacket::TypeTag::RECORD_END = "RE\0";
const char* EmotiBitPacket::TypeTag::MODE_NORMAL_POWER = "MN\0";				// Stops sending data timestamping should be accurate
const char* EmotiBitPacket::TypeTag::MODE_LOW_POWER = "ML\0";				// Stops sending data timestamping should be accurate
const char* EmotiBitPacket::TypeTag::MODE_MAX_LOW_POWER = "MM\0";		// Stops sending data timestamping accuracy drops
const char* EmotiBitPacket::TypeTag::MODE_WIRELESS_OFF = "MO\0";				// Stops sending data timestamping should be accurate
const char* EmotiBitPacket::TypeTag::MODE_HIBERNATE = "MH\0";				// Full shutdown of all operation
const char* EmotiBitPacket::TypeTag::EMOTIBIT_DISCONNECT = "ED\0";
const char* EmotiBitPacket::TypeTag::SERIAL_DATA_ON = "S+\0";
const char* EmotiBitPacket::TypeTag::SERIAL_DATA_OFF = "S-\0";
// Advertising TypeTags
const char* EmotiBitPacket::TypeTag::PING = "PN\0";
const char* EmotiBitPacket::TypeTag::PONG = "PO\0";
const char* EmotiBitPacket::TypeTag::HELLO_EMOTIBIT = "HE\0";
const char* EmotiBitPacket::TypeTag::HELLO_HOST = "HH\0";
const char* EmotiBitPacket::TypeTag::EMOTIBIT_CONNECT = "EC\0";
// WiFi Credential management TypeTags
const char* EmotiBitPacket::TypeTag::WIFI_ADD = "WA\0";
const char* EmotiBitPacket::TypeTag::WIFI_DELETE = "WD\0";

//Information Exchange TypeTags
const char* EmotiBitPacket::TypeTag::LIST = "LS\0";

const char* EmotiBitPacket::PayloadLabel::CONTROL_PORT = "CP\0";
const char* EmotiBitPacket::PayloadLabel::DATA_PORT = "DP\0";
const char* EmotiBitPacket::PayloadLabel::DEVICE_ID = "DI\0";
const char* EmotiBitPacket::PayloadLabel::RECORDING_STATUS = "RS\0";
const char* EmotiBitPacket::PayloadLabel::POWER_STATUS = "PS\0";
const char* EmotiBitPacket::PayloadLabel::LSL_MARKER_RX_TIMESTAMP = "LR\0";
const char* EmotiBitPacket::PayloadLabel::LSL_MARKER_SRC_TIMESTAMP = "LM\0";
const char* EmotiBitPacket::PayloadLabel::LSL_LOCAL_CLOCK_TIMESTAMP = "LC\0";
const char* EmotiBitPacket::PayloadLabel::LSL_MARKER_DATA = "LD\0";



const char EmotiBitPacket::PACKET_DELIMITER_CSV = '\n';
//const uint8_t nAperiodicTypeTags = 2;
//const uint8_t nUserMessagesTypeTags = 1;
// Defining TypeTag groups
const char* const EmotiBitPacket::TypeTagGroups::APERIODIC[] = 
		{
		EmotiBitPacket::TypeTag::HEART_RATE, 
		EmotiBitPacket::TypeTag::INTER_BEAT_INTERVAL, 
		EmotiBitPacket::TypeTag::SKIN_CONDUCTANCE_RESPONSE_AMPLITUDE,
		EmotiBitPacket::TypeTag::SKIN_CONDUCTANCE_RESPONSE_RISE_TIME
		};
uint8_t EmotiBitPacket::TypeTagGroups::NUM_APERIODIC = sizeof(EmotiBitPacket::TypeTagGroups::APERIODIC) / sizeof(EmotiBitPacket::TypeTagGroups::APERIODIC[0]);
const char* const EmotiBitPacket::TypeTagGroups::USER_MESSAGES[] = {EmotiBitPacket::TypeTag::USER_NOTE};
uint8_t EmotiBitPacket::TypeTagGroups::NUM_USER_MESSAGES = sizeof(EmotiBitPacket::TypeTagGroups::USER_MESSAGES) / sizeof(EmotiBitPacket::TypeTagGroups::USER_MESSAGES[0]);
//vector<string> EmotiBitPacket::TypeTag::APERIODIC.push_back(EmotiBitPacket::TypeTag::DATA_CLIPPING);
const char* const EmotiBitPacket::TypeTagGroups::COMPOSITE_PAYLOAD[] =
{
	EmotiBitPacket::TypeTag::TIMESTAMP_CROSS_TIME,
	EmotiBitPacket::TypeTag::LSL_MARKER
};
uint8_t EmotiBitPacket::TypeTagGroups::NUM_COMPOSITE_PAYLOAD = sizeof(EmotiBitPacket::TypeTagGroups::COMPOSITE_PAYLOAD) / sizeof(EmotiBitPacket::TypeTagGroups::COMPOSITE_PAYLOAD[0]);
#ifdef ARDUINO
	const String EmotiBitPacket::TIMESTAMP_STRING_FORMAT = "%Y-%m-%d_%H-%M-%S-%f";
#else
	const string EmotiBitPacket::TIMESTAMP_STRING_FORMAT = "%Y-%m-%d_%H-%M-%S-%f";
#endif


#ifdef ARDUINO
//static String getTypeTag(const Header &h);
#else
//string EmotiBitPacket::getTypeTag(const Header h) {
//}
#endif

//EmotiBitPacket::EmotiBitPacket() {
//const char* EmotiBitPacket::typeTags[(uint8_t)EmotiBitPacket::PacketType::length];
//EmotiBitPacket::typeTags[(uint8_t)EmotiBitPacket::PacketType::EDA] = &EmotiBitPacket::TypeTag::EDA;
	//typeTags[PacketType::EDL] = EmotiBitPacket::TypeTag::EDL;
	//typeTags[PacketType::EDR] = EmotiBitPacket::TypeTag::EDR;
	//typeTags[PacketType::PPG_INFRARED] = EmotiBitPacket::TypeTag::PPG_INFRARED;
	//typeTags[PacketType::PPG_RED] = EmotiBitPacket::TypeTag::PPG_RED;
	//typeTags[PacketType::PPG_GREEN] = EmotiBitPacket::TypeTag::PPG_GREEN;
	//typeTags[PacketType::TEMPERATURE_0] = EmotiBitPacket::TypeTag::TEMPERATURE_0;
	//typeTags[PacketType::THERMISTOR] = EmotiBitPacket::TypeTag::THERMISTOR;
	//typeTags[PacketType::HUMIDITY_0] = EmotiBitPacket::TypeTag::HUMIDITY_0;
	//typeTags[PacketType::ACCELEROMETER_X] = EmotiBitPacket::TypeTag::ACCELEROMETER_X;
	//typeTags[PacketType::ACCELEROMETER_Y] = EmotiBitPacket::TypeTag::ACCELEROMETER_Y;
	//typeTags[PacketType::ACCELEROMETER_Z] = EmotiBitPacket::TypeTag::ACCELEROMETER_Z;
	//typeTags[PacketType::GYROSCOPE_X] = EmotiBitPacket::TypeTag::GYROSCOPE_X;
	//typeTags[PacketType::GYROSCOPE_Y] = EmotiBitPacket::TypeTag::GYROSCOPE_Y;
	//typeTags[PacketType::GYROSCOPE_Z] = EmotiBitPacket::TypeTag::GYROSCOPE_Z;
	//typeTags[PacketType::MAGNETOMETER_X] = EmotiBitPacket::TypeTag::MAGNETOMETER_X;
	//typeTags[PacketType::MAGNETOMETER_Y] = EmotiBitPacket::TypeTag::MAGNETOMETER_Y;
	//typeTags[PacketType::MAGNETOMETER_Z] = EmotiBitPacket::TypeTag::MAGNETOMETER_Z;
	//typeTags[PacketType::BATTERY_VOLTAGE] = EmotiBitPacket::TypeTag::BATTERY_VOLTAGE;
	//typeTags[PacketType::BATTERY_PERCENT] = EmotiBitPacket::TypeTag::BATTERY_PERCENT;
	//typeTags[PacketType::DATA_CLIPPING] = EmotiBitPacket::TypeTag::DATA_CLIPPING;
	//typeTags[PacketType::DATA_OVERFLOW] = EmotiBitPacket::TypeTag::DATA_OVERFLOW;
	//typeTags[PacketType::SD_CARD_PERCENT] = EmotiBitPacket::TypeTag::SD_CARD_PERCENT;
	//typeTags[PacketType::RESET] = EmotiBitPacket::TypeTag::RESET;
	//typeTags[PacketType::GPS_LATLNG] = EmotiBitPacket::TypeTag::GPS_LATLNG;
	//typeTags[PacketType::GPS_SPEED] = EmotiBitPacket::TypeTag::GPS_SPEED;
	//typeTags[PacketType::GPS_BEARING] = EmotiBitPacket::TypeTag::GPS_BEARING;
	//typeTags[PacketType::GPS_ALTITUDE] = EmotiBitPacket::TypeTag::GPS_ALTITUDE;
	//typeTags[PacketType::TIMESTAMP_LOCAL] = EmotiBitPacket::TypeTag::TIMESTAMP_LOCAL;
	//typeTags[PacketType::TIMESTAMP_UTC] = EmotiBitPacket::TypeTag::TIMESTAMP_UTC;
	//typeTags[PacketType::RECORD_BEGIN] = EmotiBitPacket::TypeTag::RECORD_BEGIN;
	//typeTags[PacketType::RECORD_END] = EmotiBitPacket::TypeTag::RECORD_END;
	//typeTags[PacketType::USER_NOTE] = EmotiBitPacket::TypeTag::USER_NOTE;
	//typeTags[PacketType::MODE_HIBERNATE] = EmotiBitPacket::TypeTag::MODE_HIBERNATE;
	//typeTags[PacketType::ACK] = EmotiBitPacket::TypeTag::ACK;
	//typeTags[PacketType::HELLO_EMOTIBIT] = EmotiBitPacket::TypeTag::HELLO_EMOTIBIT;
	//typeTags[PacketType::REQUEST_DATA] = EmotiBitPacket::TypeTag::REQUEST_DATA;
	//typeTags[PacketType::PING] = EmotiBitPacket::TypeTag::PING;
	//typeTags[PacketType::PONG] = EmotiBitPacket::TypeTag::PONG;
//}


int16_t EmotiBitPacket::getHeader(const String &packet, Header &packetHeader) 
{
	int16_t dataStartChar = 0;

	int16_t commaN;
	int16_t commaN1;
	// timestamp
	commaN = 0;
	commaN1 = packet.indexOf(PAYLOAD_DELIMITER, commaN);
	if (commaN1 == -1) return MALFORMED_HEADER;
	packetHeader.timestamp = packet.substring(commaN, commaN1).toInt();
	// packet_number
	commaN = commaN1 + 1;
	commaN1 = packet.indexOf(PAYLOAD_DELIMITER, commaN);
	if (commaN1 == -1) return MALFORMED_HEADER;
	packetHeader.packetNumber = packet.substring(commaN, commaN1).toInt();
	// data_length
	commaN = commaN1 + 1;
	commaN1 = packet.indexOf(PAYLOAD_DELIMITER, commaN);
	if (commaN1 == -1) return MALFORMED_HEADER;
	packetHeader.dataLength = packet.substring(commaN, commaN1).toInt();
	// typetag
	commaN = commaN1 + 1;
	commaN1 = packet.indexOf(PAYLOAD_DELIMITER, commaN);
	if (commaN1 == -1) return MALFORMED_HEADER;
#ifdef ARDUINO
	// ToDo: Handle string = String more gracefully
	packetHeader.typeTag = packet.substring(commaN, commaN1);
#else
	packetHeader.typeTag = packet.substring(commaN, commaN1).str;
#endif
	// protocol_version
	commaN = commaN1 + 1;
	commaN1 = packet.indexOf(PAYLOAD_DELIMITER, commaN);
	if (commaN1 == -1) return MALFORMED_HEADER;
	packetHeader.protocolVersion = packet.substring(commaN, commaN1).toInt();
	// data_reliability
	commaN = commaN1 + 1;
	commaN1 = packet.indexOf(PAYLOAD_DELIMITER, commaN);
	if (commaN1 == -1) 
	{
		// handle case when no ,[data] exists
		commaN1 = packet.length();
		dataStartChar = NO_PACKET_DATA;
	}
	else
	{
		//dataStartChar = 11111;
		dataStartChar = commaN1 + 1;
	}
	packetHeader.dataReliability = packet.substring(commaN, commaN1).toInt();

	return dataStartChar;
}
#ifndef ARDUINO
bool EmotiBitPacket::getHeader(const vector<string>& packet, Header &packetHeader) 
{

	if (packet.size() >= EmotiBitPacket::headerLength) {

		try {

			if (packet.at(0) != "") {
				packetHeader.timestamp = stoi(packet.at(0));
			}
			else return false;

			if (packet.at(1) != "") {
				//uint16_t tempPacketNumber = stoi(packet.at(1));
				//if (tempPacketNumber - packetHeader.packetNumber > 1) {
				//	cout << "**  Missed packet: " << packetHeader.packetNumber << EmotiBitPacket::PAYLOAD_DELIMITER << tempPacketNumber << "**" << endl;
				//}
				// ToDo: Figure out a way to deal with multiple packets of each number
				//packetHeader.packetNumber = tempPacketNumber;
				packetHeader.packetNumber = stoi(packet.at(1));
			}
			else return false;

			if (packet.at(2) != "") {
				packetHeader.dataLength = stoi(packet.at(2));
			}
			else return false;

			if (packet.at(3) != "") {
				packetHeader.typeTag = packet.at(3);
			}
			else return false;

			if (packet.at(4) != "") {
				packetHeader.protocolVersion = stoi(packet.at(4));
			}
			else return false;

			if (packet.at(5) != "") {
				packetHeader.dataReliability = stoi(packet.at(5));
			}
			else return false;

		}
		catch (exception e) {
			// non-specific exception.. probably malformed packet garbage
			return false;
		}

		if (packet.size() < (size_t) EmotiBitPacket::headerLength + packetHeader.dataLength) {
			//malformedMessages++;
			//cout << "**** MALFORMED MESSAGE " << malformedMessages << ", " << messageLen << " ****" << endl;
			return false;
		}

	}
	else {
		return false;
	}
	return true;
}
#endif

#ifdef ARDUINO
EmotiBitPacket::Header EmotiBitPacket::createHeader(const String &typeTag, uint32_t timestamp, uint16_t packetNumber, uint16_t dataLength, uint8_t protocolVersion, uint8_t dataReliability)
#else
EmotiBitPacket::Header EmotiBitPacket::createHeader(const string &typeTag, uint32_t timestamp, uint16_t packetNumber, uint16_t dataLength, uint8_t protocolVersion, uint8_t dataReliability)
#endif
{
	EmotiBitPacket::Header header;
	header.typeTag = typeTag;
	header.timestamp = timestamp;
	header.packetNumber = packetNumber;
	header.dataLength = dataLength;
	header.protocolVersion = protocolVersion;
	header.dataReliability = dataReliability;

	return header;
}

#ifdef ARDUINO
String EmotiBitPacket::headerToString(const Header &header)
{
	String headerString;
	headerString = "";
	headerString += header.timestamp;
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += header.packetNumber;
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += header.dataLength;
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += header.typeTag;
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += header.protocolVersion;
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += header.dataReliability;
#else
string EmotiBitPacket::headerToString(const Header &header)
{
	string headerString;
	headerString = "";
	headerString += ofToString(header.timestamp);
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += ofToString(header.packetNumber);
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += ofToString(header.dataLength);
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += ofToString(header.typeTag);
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += ofToString((int)header.protocolVersion);
	headerString += EmotiBitPacket::PAYLOAD_DELIMITER;
	headerString += ofToString((int)header.dataReliability);
#endif
	//createPacketHeader(tempHeader, timestamp, typeTag, dataLen);
	return headerString;
}



int16_t EmotiBitPacket::getPacketElement(const String& packet, String& element, uint16_t startChar)
{
	int16_t nextStartChar = -1;

	int16_t commaN1 = packet.indexOf(PAYLOAD_DELIMITER, startChar);

	if (commaN1 != -1) 
	{
		// A following comma was found, extract element
		element = packet.substring(startChar, commaN1);
		if (packet.length() > commaN1 + 1)
		{
			nextStartChar = commaN1 + 1;
		}
	}
	else if (packet.length() > startChar + 1)
	{
		// No following comma was found, return final element of packet
		element = packet.substring(startChar, packet.length());
	}
	return nextStartChar;
}

int16_t EmotiBitPacket::getPacketKeyedValue(const String &packet, const String &key, String &value, int16_t startChar)
{
	String element;
	do
	{
		startChar = EmotiBitPacket::getPacketElement(packet, element, startChar);
		if (element.equals(key))
		{
			EmotiBitPacket::getPacketElement(packet, value, startChar);
			if (value.length() > 0) {
				return startChar;
			}
			return -1;	// return -1 if we hit the end of the packet before finding a value
		}
	} while (startChar > -1);

	return -1;	// return -1 if we hit the end of the packet before finding key
}

#ifdef ARDUINO

String EmotiBitPacket::createPacket(const String &typeTag, const uint16_t &packetNumber, const String &data, const uint16_t &dataLength, const uint8_t& protocolVersion, const uint8_t& dataReliability)
{
	// ToDo: Generalize createPacket to work across more platforms inside EmotiBitPacket
	EmotiBitPacket::Header header = EmotiBitPacket::createHeader(typeTag, millis(), packetNumber, dataLength, protocolVersion, dataReliability);
	return EmotiBitPacket::headerToString(header) + EmotiBitPacket::PAYLOAD_DELIMITER + data + EmotiBitPacket::PACKET_DELIMITER_CSV;
}
#else
int16_t EmotiBitPacket::getPacketElement(const string &packet, string &element, uint16_t startChar)
{
	// ToDo: try out a more passthrough approach to overloading
	String elementS(element);
	int16_t pos = getPacketElement(String(packet), elementS, startChar);
	element = elementS.str;
	return pos;
}
int16_t EmotiBitPacket::getPacketKeyedValue(const string &packet, const string &key, string &value, uint16_t startChar)
{
	// ToDo: try out a more passthrough approach to overloading
	String valueS(value);
	int16_t pos = getPacketKeyedValue(String(packet), String(key), valueS, startChar);
	value = valueS.str;
	return pos;
}
int16_t EmotiBitPacket::getPacketKeyedValue(const vector<string> &splitPacket, const string &key, string &value, uint16_t startIndex)
{
	for (size_t i = startIndex; i < splitPacket.size(); i++)
	{
		if (key.compare(splitPacket.at(i)) == 0)
		{
			i++;
			value = splitPacket.at(i);
			return i;
		}
	}
	return -1;
}

string EmotiBitPacket::createPacket(const string &typeTag, const uint16_t &packetNumber, const string &data, const uint16_t &dataLength, const uint8_t& protocolVersion, const uint8_t& dataReliability)
{
	// ToDo: Generalize createPacket to work across more platforms inside EmotiBitPacket
	EmotiBitPacket::Header header = EmotiBitPacket::createHeader(typeTag, ofGetElapsedTimeMillis(), packetNumber, dataLength, protocolVersion, dataReliability);
	if (dataLength == 0)
	{
		return EmotiBitPacket::headerToString(header) + EmotiBitPacket::PACKET_DELIMITER_CSV;
	}
	else
	{
		return EmotiBitPacket::headerToString(header) + EmotiBitPacket::PAYLOAD_DELIMITER + data + EmotiBitPacket::PACKET_DELIMITER_CSV;
	}
}

string EmotiBitPacket::createPacket(const string &typeTag, const uint16_t &packetNumber, const vector<string> & data, const uint8_t &protocolVersion, const uint8_t &dataReliability)
{
	// ToDo: Template data vector
	// ToDo: Generalize createPacket to work across more platforms inside EmotiBitPacket
	EmotiBitPacket::Header header = EmotiBitPacket::createHeader(typeTag, ofGetElapsedTimeMillis(), packetNumber, data.size(), protocolVersion, dataReliability);
	string packet = EmotiBitPacket::headerToString(header);
	for (string s : data)
	{
		packet += EmotiBitPacket::PAYLOAD_DELIMITER + s;
	}
	packet += EmotiBitPacket::PACKET_DELIMITER_CSV;
	return packet;
}
#endif
