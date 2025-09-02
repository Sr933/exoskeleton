#include "EmotiBitSerial.h"

#ifdef ARDUINO
bool EmotiBitSerial::parseSerialMessage(String message, String &typetag, String &payload)
{
	if(message.charAt(0) == MSG_START_CHAR)
	{
		// msg has valid start condition
		int messageDelimiterIndex = message.indexOf(MSG_TERM_CHAR);
		if(messageDelimiterIndex > 0)
		{
			// valid msg
			int payloadDelimiterIndex = message.indexOf(PAYLOAD_DELIMITER);
			if (payloadDelimiterIndex < 0 )
			{
				// No payload
				// msg of the type @TT~
				typetag = message.substring(1, messageDelimiterIndex);
				payload = "";
			}
			else
			{
				if (payloadDelimiterIndex == messageDelimiterIndex - 1)
				{
					// msg of the type @TT,~
					return false;
				}
				else if (payloadDelimiterIndex == messageDelimiterIndex - 2 && isSpace(message.charAt(payloadDelimiterIndex+1)))
				{
					// msg of the type @TT, ~
					return false;
				}
				else
				{
					// msg of the type @TT,payload~
					typetag = message.substring(1, payloadDelimiterIndex);
					payload = message.substring(payloadDelimiterIndex + 1, messageDelimiterIndex);

				}
			}
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
	// ToDo: Add a debug pre-processor guard instead of comments
	//Serial.print("typetag: ");Serial.println(typetag);
	//Serial.print("payload: ");Serial.println(payload);
	return true;
}

void EmotiBitSerial::sendMessage(String typeTag, String payload)
{
    Serial.print(EmotiBitSerial::MSG_START_CHAR);
	Serial.print(typeTag);
	if (!payload.equals(""))
	{
		Serial.print(EmotiBitSerial::PAYLOAD_DELIMITER);
		Serial.print(payload);
	}
	Serial.println(EmotiBitSerial::MSG_TERM_CHAR);
}
#endif