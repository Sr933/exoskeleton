#include "EmotiBitVariants.h"
#ifdef ARDUINO
#include "EmotiBitVersionController.h"
#endif

const char* EmotiBitVariants::EMOTIBIT_SKU_MD = "MD\0";
const char* EmotiBitVariants::EMOTIBIT_SKU_EM = "EM\0";
#ifdef ARDUINO
void printEmotiBitVariantInfo(EmotiBitVariantInfo_V1 emotibitVariantInfo)
{
	Serial.print("EmotiBit HW Version: "); Serial.println(EmotiBitVersionController::getHardwareVersion((EmotiBitVersionController::EmotiBitVersion)emotibitVariantInfo.hwVersion));
	Serial.print("EmotiBit Sku Version: "); Serial.println(emotibitVariantInfo.sku);
	if (emotibitVariantInfo.emotibitSerialNumber != UINT32_MAX)
	{
		Serial.print("EmotiBit Number: "); Serial.println(emotibitVariantInfo.emotibitSerialNumber);
	}
}
#endif