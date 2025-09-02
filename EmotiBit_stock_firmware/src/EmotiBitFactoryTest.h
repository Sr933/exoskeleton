#pragma once

#ifdef ARDUINO
	#include <Arduino.h>
#else
	// ToDo: Remove OF dependency for ofToString()
	#include "ofMain.h"
#endif // !ARDUINO
#include "EmotiBitVariants.h"
#include "EmotiBitSerial.h"

#ifdef ARDUINO
struct Barcode
{
	String rawCode = "\0";
	String sku = "\0";
	String hwVersion = "\0";
	String emotibitSerialNumber = "\0";
};
#endif

class EmotiBitFactoryTest{
public:
	class TypeTag
	{
	public:
				// Factory Test Typetags
		static const char* EMOTIBIT_VERSION;
		static const char* FIRMWARE_VERSION;
		static const char* EMOTIBIT_BARCODE;
		static const char* EMOTIBIT_SERIAL_NUMBER;
		static const char* EMOTIBIT_SKU_TYPE;
		static const char* I2C_COMM_INIT;
		static const char* FLASH;
		static const char* SERIAL_NUMBER_WRITE;
		static const char* LED_CONTROLLER;
		static const char* PPG_SENSOR;
		static const char* IMU_ID;
		static const char* ACCEL_GYRO;
		static const char* MAGNETOMETER;
		static const char* TEMP_SENSOR;
		static const char* THERMOPILE;
		static const char* ADC_INIT;
		static const char* SD_CARD;
		static const char* WIFI;
		static const char* SETUP_COMPLETE;
		static const char* AVDD_VOLTAGE;
		//static const char* AVDD_PASS;
		static const char* DVDD_VOLTAGE;
		//static const char* DVDD_PASS;
		static const char* TOTAL_CURRENT;
		static const char* WIFI_OFF_CURRENT;
		static const char* HIBERNATE_CURRENT;
		static const char* TEST_RESULT_DELIMITER;
		static const char* TEST_TYPE_DELIMITER;
		static const char* TEST_PASS;
		static const char* TEST_FAIL;
		static const char* LED_RED_ON;
		static const char* LED_RED_OFF;
		static const char* LED_BLUE_ON;
		static const char* LED_BLUE_OFF;
		static const char* LED_YELLOW_ON;
		static const char* LED_YELLOW_OFF;
		static const char* NULL_VAL;
		static const char* VERSION_VALIDATION;
		static const char* SKU_VALIDATION;
		static const char* EDA_CALIBRATION_VALUES;
		static const char* EDA_CALIBRATION_ACK;
		static const char* EMOTIBIT_READY;
		static const char* EMOTIBIT_STORAGE;
	};

	// definitions that can exist outside the scope of factory test are defined in EmotiBitSerial. 
	static const char BARCODE_DELIMITER = '-';
	static const char INIT_FACTORY_TEST = EmotiBitSerial::Inputs::FACTORY_TEST_MODE;//'F';
	static const char FIRMWARE_DELIMITER = '.';
	#ifdef ARDUINO
	static void updateOutputString(String &output, const char* testType, const char* result);
	static void sendMessage(String typeTag, String payload = "");
	static void parseBarcode(Barcode* barcode);
	static bool validateVersionEstimate(String barcode, String estimate);
	static void convertBarcodeToVariantInfo(Barcode barcode, EmotiBitVariantInfo_V1 &emotiBitVariantInfo);
#else
	static string createPacket(string typeTag, string payload = "");
#endif
};

