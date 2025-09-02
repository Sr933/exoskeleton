#include "EmotiBitFactoryTest.h"
#ifdef ARDUINO
#include "EmotiBitVersionController.h"
#endif
// Factory Test Typetags
const char* EmotiBitFactoryTest::TypeTag::EMOTIBIT_VERSION = "EV\0";
const char* EmotiBitFactoryTest::TypeTag::FIRMWARE_VERSION = "FV\0";
const char* EmotiBitFactoryTest::TypeTag::EMOTIBIT_BARCODE = "BC\0";
const char* EmotiBitFactoryTest::TypeTag::EMOTIBIT_SERIAL_NUMBER = "EN\0";
const char* EmotiBitFactoryTest::TypeTag::EMOTIBIT_SKU_TYPE = "ST\0";
const char* EmotiBitFactoryTest::TypeTag::I2C_COMM_INIT = "TW\0";
const char* EmotiBitFactoryTest::TypeTag::FLASH = "FL\0";
const char* EmotiBitFactoryTest::TypeTag::SERIAL_NUMBER_WRITE = "SW\0";
const char* EmotiBitFactoryTest::TypeTag::LED_CONTROLLER = "LC\0";
const char* EmotiBitFactoryTest::TypeTag::PPG_SENSOR = "PG\0";
const char* EmotiBitFactoryTest::TypeTag::IMU_ID = "II\0";
const char* EmotiBitFactoryTest::TypeTag::ACCEL_GYRO = "AG\0";
const char* EmotiBitFactoryTest::TypeTag::MAGNETOMETER = "MG\0";
const char* EmotiBitFactoryTest::TypeTag::TEMP_SENSOR = "TE\0";
const char* EmotiBitFactoryTest::TypeTag::THERMOPILE = "TI\0";
const char* EmotiBitFactoryTest::TypeTag::ADC_INIT = "AD\0";
const char* EmotiBitFactoryTest::TypeTag::SD_CARD = "SD\0";
const char* EmotiBitFactoryTest::TypeTag::WIFI = "WF\0";
const char* EmotiBitFactoryTest::TypeTag::SETUP_COMPLETE = "SC\0";
const char* EmotiBitFactoryTest::TypeTag::AVDD_VOLTAGE = "AV\0";
const char* EmotiBitFactoryTest::TypeTag::DVDD_VOLTAGE = "DV\0";
const char* EmotiBitFactoryTest::TypeTag::TOTAL_CURRENT = "TC\0";
const char* EmotiBitFactoryTest::TypeTag::WIFI_OFF_CURRENT = "WC\0";
const char* EmotiBitFactoryTest::TypeTag::HIBERNATE_CURRENT = "HC\0";
const char* EmotiBitFactoryTest::TypeTag::TEST_RESULT_DELIMITER = ":\0";
const char* EmotiBitFactoryTest::TypeTag::TEST_TYPE_DELIMITER = "+\0";
const char* EmotiBitFactoryTest::TypeTag::LED_RED_ON = "+R\0";
const char* EmotiBitFactoryTest::TypeTag::LED_RED_OFF = "-R\0";
const char* EmotiBitFactoryTest::TypeTag::LED_BLUE_ON = "+B\0";
const char* EmotiBitFactoryTest::TypeTag::LED_BLUE_OFF = "-B\0";
const char* EmotiBitFactoryTest::TypeTag::LED_YELLOW_ON = "+Y\0";
const char* EmotiBitFactoryTest::TypeTag::LED_YELLOW_OFF = "-Y\0";
const char* EmotiBitFactoryTest::TypeTag::NULL_VAL = "\0";
const char* EmotiBitFactoryTest::TypeTag::TEST_PASS = "PASS\0";
const char* EmotiBitFactoryTest::TypeTag::TEST_FAIL = "FAIL\0";
const char* EmotiBitFactoryTest::TypeTag::VERSION_VALIDATION = "VV\0";
const char* EmotiBitFactoryTest::TypeTag::SKU_VALIDATION = "SV\0";
const char* EmotiBitFactoryTest::TypeTag::EDA_CALIBRATION_VALUES = "EC\0";
const char* EmotiBitFactoryTest::TypeTag::EDA_CALIBRATION_ACK = "EK\0";
const char* EmotiBitFactoryTest::TypeTag::EMOTIBIT_READY = "ER\0";
const char* EmotiBitFactoryTest::TypeTag::EMOTIBIT_STORAGE = "ES\0";

#ifdef ARDUINO
void EmotiBitFactoryTest::updateOutputString(String &output, const char* testType, const char* result)
{
	output += testType;
	output += TypeTag::TEST_RESULT_DELIMITER;
	output += result;
	if (testType != TypeTag::SETUP_COMPLETE)
	{
		output += TypeTag::TEST_TYPE_DELIMITER;
	}
	else
	{
		output += EmotiBitSerial::MSG_TERM_CHAR;
	}
}

void EmotiBitFactoryTest::sendMessage(String typeTag, String payload)
{
	EmotiBitSerial::sendMessage(typeTag, payload);
}

	// Parses the barcode 
void EmotiBitFactoryTest::parseBarcode(Barcode* barcode)
{
	String rawBarcode;
	rawBarcode = barcode->rawCode;
	barcode->sku = rawBarcode.substring(0, rawBarcode.indexOf(BARCODE_DELIMITER));
	rawBarcode = rawBarcode.substring(rawBarcode.indexOf(BARCODE_DELIMITER) + 1);
	barcode->hwVersion = rawBarcode.substring(0, rawBarcode.indexOf(BARCODE_DELIMITER));
	barcode->emotibitSerialNumber = rawBarcode.substring(rawBarcode.indexOf(BARCODE_DELIMITER) + 1);
}

bool EmotiBitFactoryTest::validateVersionEstimate(String barcode, String estimate)
{
	// remove the leading "V" in version
	barcode.remove(barcode.indexOf(EmotiBitVariants::HARDWARE_VERSION_PREFIX),1);
	estimate.remove(estimate.indexOf(EmotiBitVariants::HARDWARE_VERSION_PREFIX),1);
	if (barcode.toInt() == estimate.toInt())
		return true;
	else
		return false;
}

void EmotiBitFactoryTest::convertBarcodeToVariantInfo(Barcode barcode, EmotiBitVariantInfo_V1 &emotiBitVariantInfo)
{
	for (uint8_t i = 0; i < (uint8_t)EmotiBitVersionController::EmotiBitVersion::length; i++)
	{
		String hwVersionStr = EmotiBitVersionController::getHardwareVersion((EmotiBitVersionController::EmotiBitVersion)i);
		String barcodeHwVersion = barcode.hwVersion;
		hwVersionStr.remove(hwVersionStr.indexOf(EmotiBitVariants::HARDWARE_VERSION_PREFIX), 1);
		barcodeHwVersion.remove(barcodeHwVersion.indexOf(EmotiBitVariants::HARDWARE_VERSION_PREFIX), 1);
		if (barcodeHwVersion.toInt() == hwVersionStr.toInt())
			emotiBitVariantInfo.hwVersion = i;
	}
	
	if (barcode.sku.equals(EmotiBitVariants::EMOTIBIT_SKU_EM))
		strncpy(emotiBitVariantInfo.sku, EmotiBitVariants::EMOTIBIT_SKU_EM,EmotiBitVariants::EMOTIBIT_SKU_LENGTH);
	else if(barcode.sku.equals(EmotiBitVariants::EMOTIBIT_SKU_MD))
		strncpy(emotiBitVariantInfo.sku, EmotiBitVariants::EMOTIBIT_SKU_MD, EmotiBitVariants::EMOTIBIT_SKU_LENGTH);

	emotiBitVariantInfo.emotibitSerialNumber = barcode.emotibitSerialNumber.toInt();
}
#else


string EmotiBitFactoryTest::createPacket(string typeTag, string payload)
{
	string s = "";
	s += EmotiBitSerial::MSG_START_CHAR;
	s += typeTag;
	if (payload.length() > 0) {
		s += EmotiBitSerial::PAYLOAD_DELIMITER;
		s += payload;
	}
	s += EmotiBitSerial::MSG_TERM_CHAR;
	return s;
}
#endif