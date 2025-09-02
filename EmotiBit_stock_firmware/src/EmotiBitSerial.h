#pragma once
#ifdef ARDUINO
#include <Arduino.h>
#endif

class EmotiBitSerial
{
public:
#ifdef ARDUINO
    static bool parseSerialMessage(String message, String &typetag, String &payload);
    static void sendMessage(String typeTag, String payload = "");
    
#endif
    static const char MSG_START_CHAR = '@';
	static const char MSG_TERM_CHAR = '~';
	static const char PAYLOAD_DELIMITER = ',';

    /*!
    * Inputs that can change EmotiBit functionality. These prompts can be made by a user or by a software.
    */
    struct Inputs
    {
        static const char ADC_CORRECTION_MODE = 'A';  ///< enter ADC correction mode. Backward compatibility to emotibit v2/v3 
        static const char RESET = 'R';  ///< resets MCU
        static const char CRED_UPDATE = 'C';  ///< Enters config file update mode.
        static const char DEBUG_MODE = 'D';   ///< Enters Debug mode
        static const char FACTORY_TEST_MODE = 'F';  ///< Enters Factory Test mode
    };
};