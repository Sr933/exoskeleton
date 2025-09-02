/**************************************************************************/
/*!
		@file     EmotiBitEdaCalibration.h
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

#pragma once

#ifdef ARDUINO
#include <String.h>
#include <WString.h>
#include <stdint.h>
#endif

#ifndef ARDUINO
#include <string>
#include <vector>
//#include "ofMain.h"	// ToDo: sort out chrono overload to remove this dependency
//using namespace std;
#endif

class EmotiBitEdaCalibration
{
public:

	static const uint8_t V0 = 0;
	// Calibration Raw Values V0
	struct RawValues_V0
	{
		float edl0R;
		float edl10K;
		float edl100K;
		float edl1M;
		float edl10M;
		float edrAvg;
	};

	static const uint8_t V1 = 1;
	// Calibration Raw Values V1
	struct RawValues_V1
	{
		float edl0R;
		float edl10K;
		float edl100K;
		float edl1M;
		float edl10M;
		float edr[5];
	};

	struct CalibPair
	{
		float res;
		float adcVal;
	};

	static const uint8_t V2 = 2;
	static const unsigned int nCalibVals_V2 = 5;
	// Calibration Raw Values V2
	struct RawValues_V2
	{
		unsigned int nVals = nCalibVals_V2;
		CalibPair vals[nCalibVals_V2];
	};


	/*!
	@brief Calculate EDA calibration values
	@param rawVals Struct with raw values to calculate derivatives
	@param edaTransformSlope Output slope to convert ADC bits/volts to resistance
	@param edaTransformIntercept Output offset to convert ADC bits/volts to resistance
	@return true if calibration calculation succeeded
	*/
	static bool calculate(const RawValues_V2 &rawVals, float &edaTransformSlope, float &edaTransformIntercept);
	/*!
	@brief Calculate EDA calibration values
	@param rawVals Struct with raw values to calculate derivatives
	@param vRef1 Output calibration constant
	@param vRef2 Output calibration constant
	@param rfeedback Output calibration constant
	@return true if calibration calculation succeeded
	*/
	static bool calculate(const RawValues_V0 &rawVals, float &vRef1, float &vRef2, float &rfeedback);

#ifdef ARDUINO
	/*!
		@brief unpack a packet string into a packet version and rawVals struct
		@param edaCalibPacket Packet string
		@param payloadVersion Output version of the rawVals payload
		@param rawVal Output struct of raw calibration values
		@return true if successful
	*/
	static bool unpackCalibPacket(const String &edaCalibPacket, uint8_t &payloadVersion, RawValues_V2 &rawVals);

	/*!
	@brief Serial.print contents of RawValues structs
	*/
	static void print(const RawValues_V0 &rawVals);
	static void print(const RawValues_V2 &rawVals);

#endif // ARDUINO

#ifndef ARDUINO
	// ToDo: Consider templating createCalibPacket
	/*!
		@brief pack a RawValues_V2 struct into a string for serial communication
		@param payloadVersion Version of the included rawVals
		@param rawVals Struct of the raw calibration values
		@return string with csv calibration packet data
	*/
	static std::string createCalibPacket(int payloadVersion, RawValues_V2 rawVals);

	static std::vector<float> getAdcVals(RawValues_V2 rawVals);
#endif // !ARDUINO
};