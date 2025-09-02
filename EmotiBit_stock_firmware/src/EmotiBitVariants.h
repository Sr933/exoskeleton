#pragma once
#ifdef ARDUINO
#include <Arduino.h>
#endif
/*!
	@brief Stores the Information about details that make up different variants
*/
// ToDo: Someday, the Hardware struct will also move here. That will expand access to the Software side

class EmotiBitVariants
{
public:

	static const char* EMOTIBIT_SKU_MD;
	static const char* EMOTIBIT_SKU_EM;
	static const char HARDWARE_VERSION_PREFIX = 'V';
	static const int EMOTIBIT_SKU_LENGTH = 3;
	static const int BARCODE_SERIAL_NUM_LENGTH = 7;  // Each EmotiBit has a 7 digit serial number
};

enum class EmotiBitVariantDataFormat {
	V0 = 0,
	V1 = 1,
	length
};

#ifdef ARDUINO
struct EmotiBitVariantInfo_V0
{
	uint8_t hwVersion;
};

/*!
	@brief Struct to store the HW, SKU and emotiBit number.
*/
struct EmotiBitVariantInfo_V1 {
	uint32_t emotibitSerialNumber;
	char sku[EmotiBitVariants::EMOTIBIT_SKU_LENGTH];
	uint8_t hwVersion;
};


/*!
	@brief Prints all the information about the EmotiBit variant.
	@param emotibitVariantInfo Takes in the stuct EmotiBitVariantInfo
*/
void printEmotiBitVariantInfo(EmotiBitVariantInfo_V1 emotibitVariantInfo);
#endif

