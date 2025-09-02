/**************************************************************************/
/*!
		@file     EmotiBitEdaCalibration.cpp
		@author   Sean Montgomery (EmotiBit)

		@mainpage EDA Calibration helper for EmotiBit

		@section intro_sec Introduction

		This is a library to help with EDA calibration for EmotiBit.

		EmotiBit invests time and resources providing this open source code,
		please support EmotiBit and open-source hardware by purchasing
		products from EmotiBit!

		@section author Author

		Written by Sean Montgomery for EmotiBit.

		@section  HISTORY

		v1.0  - First release

		@section license License

		BSD license, all text here must be included in any redistribution
*/
/**************************************************************************/


#include "EmotiBitEdaCalibration.h"
#include "EmotiBitFactoryTest.h"
#include "EmotiBitPacket.h"


bool EmotiBitEdaCalibration::calculate(const RawValues_V0 &rawVals, float &vRef1, float &vRef2, float &rfeedback)
{
	vRef1 = rawVals.edl0R;
	vRef2 = rawVals.edrAvg;

	// using 1M to find Rfeedback
	rfeedback = (1000000.f / ((rawVals.edl1M / vRef1) - 1));
	return true;
}

bool EmotiBitEdaCalibration::calculate(const RawValues_V2 &rawVals, float &edaTransformSlope, float &edaTransformIntercept)
{
	// ToDo: Consider optimizing calculation for maintaining float resolution
	float meanRes = 0;
	float meanAdcVal = 0;
	uint8_t nValsConsidered = 0;
	// Skip first and last rawVals because they were empirically 
	// determined to distort the best fit
	for (size_t i = 1; i < rawVals.nVals - 1; i++)
	{
		meanRes += rawVals.vals[i].res;
		meanAdcVal += rawVals.vals[i].adcVal;
		nValsConsidered++;
	}
	meanRes /= nValsConsidered;
	meanAdcVal /= nValsConsidered;

	float normRes;
	float normAdcVal;
	float normResMultNormAdcVal;
	float sumNormResMultNormAdcVal = 0;
	float normResSquare;
	float sumNormResSquare = 0;

	for (size_t i = 1; i < rawVals.nVals - 1; i++)
	{
		normRes = rawVals.vals[i].res - meanRes;
		normAdcVal = rawVals.vals[i].adcVal - meanAdcVal;
		normResMultNormAdcVal = normRes * normAdcVal;
		sumNormResMultNormAdcVal += normResMultNormAdcVal;
		normResSquare = pow(normRes, 2.f);
		sumNormResSquare += normResSquare;
	}

	edaTransformSlope = sumNormResSquare / sumNormResMultNormAdcVal;
	edaTransformIntercept = meanRes - meanAdcVal * edaTransformSlope;

	return true;
}

#ifdef ARDUINO
bool EmotiBitEdaCalibration::unpackCalibPacket(const String &edaCalibPacket, uint8_t &payloadVersion, RawValues_V2 &rawVals)
{

	String element;
	int16_t nextStartChar;

	nextStartChar = EmotiBitPacket::getPacketElement(edaCalibPacket, element, 0);
	if (nextStartChar == -1) return false; // error: ran out of packet elements before nVals
	if (element.equals(String(EmotiBitFactoryTest::TypeTag::EDA_CALIBRATION_VALUES)))
	{
		nextStartChar = EmotiBitPacket::getPacketElement(edaCalibPacket, element, nextStartChar);
		if (nextStartChar == -1) return false; // error: ran out of packet elements before nVals
		payloadVersion = element.toInt();
		if (payloadVersion != V2) return false; // error: payloadVersion mismatch with fn signature
		//Serial.println(payloadVersion);

		nextStartChar = EmotiBitPacket::getPacketElement(edaCalibPacket, element, nextStartChar);
		if (nextStartChar == -1) return false; // error: ran out of packet elements before nVals
		rawVals.nVals = element.toInt();
		//Serial.println(rawVals.nVals);

		for (size_t i = 0; i < rawVals.nVals; i++)
		{
			nextStartChar = EmotiBitPacket::getPacketElement(edaCalibPacket, element, nextStartChar);
			rawVals.vals[i].res = element.toFloat();
			//Serial.println(rawVals.vals[i].res);
			if (nextStartChar == -1) return false; // error: ran out of packet elements before nVals

			nextStartChar = EmotiBitPacket::getPacketElement(edaCalibPacket, element, nextStartChar);
			rawVals.vals[i].adcVal = element.toFloat();
			//Serial.println(rawVals.vals[i].adcVal);
			if (i < rawVals.nVals - 1 && nextStartChar == -1) return false; // error: ran out of packet elements before nVals
		}
		return true;
	}
	else
	{
		return false;
	}
}

void EmotiBitEdaCalibration::print(const RawValues_V0 &rawVals)
{
	Serial.print("0R, \t");
	Serial.println(rawVals.edl0R);
	Serial.print("10K, \t");
	Serial.println(rawVals.edl10K);
	Serial.print("1M, \t");
	Serial.println(rawVals.edl1M);
	Serial.print("10M, \t");
	Serial.println(rawVals.edl10M);
	Serial.print("EDR, \t");
	Serial.println(rawVals.edrAvg);
}

void EmotiBitEdaCalibration::print(const RawValues_V2 &rawVals)
{
	for (uint8_t i = 0; i < rawVals.nVals; i++)
	{
		if (rawVals.vals[i].res < 1000) 
		{
			Serial.print(rawVals.vals[i].res, 0);
			Serial.print("R");
		}
		else if (rawVals.vals[i].res < 1000000) 
		{
			Serial.print(rawVals.vals[i].res / 1000.f, 0);
			Serial.print("K");
		}
		else 
		{
			Serial.print(rawVals.vals[i].res / 1000000.f, 0);
			Serial.print("M");
		}
		Serial.print(", \t");
		Serial.print(rawVals.vals[i].adcVal, 6);
		Serial.println("");
	}	
}
#endif // !ARDUINO

#ifndef ARDUINO

std::string EmotiBitEdaCalibration::createCalibPacket(int payloadVersion, RawValues_V2 rawVals)
{
	std::string out;
	out += std::string(EmotiBitFactoryTest::TypeTag::EDA_CALIBRATION_VALUES) + EmotiBitPacket::PAYLOAD_DELIMITER;
	out += std::to_string(payloadVersion) + EmotiBitPacket::PAYLOAD_DELIMITER;
	out += std::to_string(rawVals.nVals) + EmotiBitPacket::PAYLOAD_DELIMITER;
	for (size_t i = 0; i < rawVals.nVals; i++)
	{
		out += std::to_string(rawVals.vals[i].res) + EmotiBitPacket::PAYLOAD_DELIMITER;
		out += std::to_string(rawVals.vals[i].adcVal);
		if (i < rawVals.nVals - 1)
		{
			out += EmotiBitPacket::PAYLOAD_DELIMITER;
		}
	}
	return out;
}

std::vector<float> EmotiBitEdaCalibration::getAdcVals(RawValues_V2 rawVals)
{
	std::vector<float> out;
	for (unsigned int i = 0; i < rawVals.nVals; i++)
	{
		out.push_back(rawVals.vals[i].adcVal);
	}
	return out;
}

#endif // !ARDUINO

