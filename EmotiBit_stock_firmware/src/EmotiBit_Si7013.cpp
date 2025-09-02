/*
CFL_Si7013.cpp - Arduino library for Si7013 humidity + temperature sensor
Library provides management of sensor measurement delays to allow activities to be performed during measurement delays
Copyright (c) 2018 Connected Future Labs. All rights reserved. http://www.connectedfuturelabs.com/
See Si7013 datasheet at https://www.silabs.com/documents/public/data-sheets/Si7013-A20.pdf

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "EmotiBit_Si7013.h"

Si7013::Si7013() {
  _humidityNew = false;
  _temperatureNew = false;
  _adcNew = false;
  _status = STATUS_IDLE;
  _delayMultiplier = 1.5f;  // Give a little extra headroom for measurements to take place
  _transactionTimeout = 100;
  _status = STATUS_IDLE;
}

bool Si7013::setup(TwoWire &wirePort, uint8_t address ) {
#ifdef DEBUG 
Serial.println("setup()");
#endif
  _i2cPort = &wirePort;
  _address = address;
  _humidityNew = false;
  _temperatureNew = false;
  _adcNew = false; 
  delay(DELAY_POWER_UP);
  _i2cPort->beginTransmission(_address);
  if (_i2cPort->endTransmission() != 0)
	  return false;
  if (getStatus() != STATUS_IDLE) return false;
	
	delay(((float)DELAY_POWER_UP) * _delayMultiplier);
  reset();
  return true;
}

bool Si7013::reset() {
#ifdef DEBUG 
Serial.println("reset()");
#endif
  if (getStatus() != STATUS_IDLE) return false;
  _i2cPort->beginTransmission(_address);
  _i2cPort->write(CMD_RESET);
  _i2cPort->endTransmission();
  delay(((float)DELAY_RESET) * _delayMultiplier);

  //// Read the registers to update the local copy of the registers
  //readRegister8(CMD_READ_REGISTER_1);
  //readRegister8(CMD_READ_REGISTER_2);
  
  // ToDo: read the register to determine the correct delays
  _humidityDelay = DELAY_HUMIDITY12BIT_TEMP14BIT;
  _adcDelay = DELAY_ADC_NORMAL;
  return true;
}

uint8_t Si7013::getStatus() {
	// ToDo: Consider using begin/endTransimission to test for success/NACK to assess ready status of sensor
#ifdef DEBUG
	Serial.print("getStatus(): "); Serial.println((int)_status);
#endif // DEBUG

	if (_status == STATUS_IDLE) {
		return _status;
	}

	uint32_t measurementTime = millis() - _measureStartTime;

#ifdef DEBUG
	Serial.print("measStartTime = ");
	Serial.print(_measureStartTime);
	Serial.print(" , millis =  ");
	Serial.println(millis());

	Serial.print("measurementTime = "); 
	Serial.print(measurementTime); 
	Serial.print(" / "); 
	Serial.println(_measurementDelay);
	//Serial.print("x");
#endif // DEBUG

  if (measurementTime > _measurementDelay) {
    if (_status == STATUS_MEASURING_HUMIDITY) {
      _humidityNew = true;
      _temperatureNew = true;
    } 
    else if (_status == STATUS_MEASURING_TEMPERATURE) {
      _temperatureNew = true;
    }
    else if (_status == STATUS_MEASURING_ADC) {
      _adcNew = true;
    }
    
    _status = STATUS_IDLE;
    
  } else {
   
  }
  return _status;
}

bool Si7013::isHumidityNew() {
#ifdef DEBUG
	Serial.println("isHumidityNew()");
#endif // DEBUG

  return _humidityNew;
}

bool Si7013::isTemperatureNew() {
#ifdef DEBUG
	Serial.println("isTemperatureNew()");
#endif // DEBUG

  return _temperatureNew;
}

bool Si7013::isAdcNew() {
#ifdef DEBUG
	Serial.println("isAdcNew()");
#endif // DEBUG

  return _adcNew;
}

int16_t Si7013::readRegister8(uint8_t reg, bool isOtpOperation) {
#ifdef DEBUG
	Serial.print("readRegister8: ");
	Serial.print("reg = ");
	Serial.println(reg);
#endif // DEBUG

  if (getStatus() == STATUS_IDLE) {

		while (_i2cPort->available() > 0) _i2cPort->read();	// discard any leftover data on the bus

    _i2cPort->beginTransmission(_address);
	if (isOtpOperation)
	{
		_i2cPort->write(CMD_OTP_READ);
	}
    _i2cPort->write(reg);
    _i2cPort->endTransmission();
		_i2cPort->requestFrom((uint8_t)_address, (uint8_t)1);
    uint32_t start = millis(); // start timeout
    while(millis()-start < _transactionTimeout) {
			if (_i2cPort->available() == 1) {

				// ToDo: set local variables from read registers
				uint8_t regVal = _i2cPort->read();
				//if (reg == CMD_READ_REGISTER_1) {
				//	_register1Value = regVal;
				//}
				//if (reg == CMD_READ_REGISTER_2) {
				//	_register2Value = regVal;
				//}

#ifdef DEBUG
				Serial.print("read val: ");
				Serial.println(regVal);
#endif

				return regVal;
			}
    }
  }
  return ERROR_READ;
}


bool Si7013::writeRegister8(uint8_t reg, uint8_t value, uint8_t mask)
{
	return _writeRegister8(reg, value, mask, false);
}

bool Si7013::writeToOtp(uint8_t reg, uint8_t value, uint8_t mask)
{
	return _writeRegister8(reg, value, mask, true);
}

bool Si7013::_writeRegister8(uint8_t reg, uint8_t value, uint8_t mask, bool isOtpOperation) {
#ifdef DEBUG
  Serial.print("writeRegister8: ");
  Serial.print("reg = ");
  Serial.print(reg);
  Serial.print(", value = ");
  Serial.print(value);
  Serial.print(", mask = ");
  Serial.println(mask);
#endif // DEBUG
  
  if (getStatus() != STATUS_IDLE) return false;

  // ToDo: consider whether storing register values would be substantially faster
  if (!isOtpOperation)
  {
	  // Read the register to preserve reserved bits
	  int16_t regValue;
	  if (reg == CMD_WRITE_REGISTER_1) {
		  regValue = readRegister8(CMD_READ_REGISTER_1);
	  }
	  if (reg == CMD_WRITE_REGISTER_2) {
		  regValue = readRegister8(CMD_READ_REGISTER_2);
	  }
	  if (reg == CMD_WRITE_REGISTER_3) {
		  regValue = readRegister8(CMD_READ_REGISTER_3);
	  }

	  if (regValue < 0) return false;

	  // Remove the reserved register bits from the mask
	  if (reg == CMD_WRITE_REGISTER_1) {
		  mask = mask & ~REG1_MASK_RSVD;
	  }
	  if (reg == CMD_WRITE_REGISTER_2) {
		  mask = mask & ~REG2_MASK_RSVD;
	  }
	  if (reg == CMD_WRITE_REGISTER_3) {
		  mask = mask & ~REG3_MASK_RSVD;
	  }
	  value = (((uint8_t) regValue) & ~mask) | value;
  }

#ifdef DEBUG
  Serial.print("write val: ");
  Serial.println(value);
#endif

  _i2cPort->beginTransmission(_address);
  // ToDo: Look into modifying OTP Access sequence as 
  // referenced in this Issue: https://github.com/EmotiBit/EmotiBit_SI7013/issues/7 
  if (isOtpOperation)
  {
	  _i2cPort->write(CMD_OTP_WRITE);
  }
  _i2cPort->write(reg);
  _i2cPort->write(value);
  _i2cPort->endTransmission();

  //if (reg == CMD_WRITE_REGISTER_1) {
	 // _register1Value = value;
  //}
  if (!isOtpOperation)
  {
	  if (reg == CMD_WRITE_REGISTER_2) {
		  // update the status of the ADC_HOLD
		  if ((value & REG2_MASK_ADC_HOLD) == REG2_VALUE_ADC_HOLD) {
			  _adcNoHold = false;
		  }
		  if ((value & REG2_MASK_ADC_HOLD) == REG2_VALUE_ADC_NO_HOLD) {
			  _adcNoHold = true;
		  }

		  //_register2Value = value;
	  }
  }
  return true;
}

bool Si7013::changeSetting(Settings setting) {
#ifdef DEBUG
	Serial.print("changeSetting: ");
	Serial.println((int) setting);
#endif // DEBUG
  
  uint8_t writeRegister;
  uint8_t mask;
  uint8_t value;
  
  if (setting == Settings::RESOLUTION_H12_T14) {
      writeRegister = CMD_WRITE_REGISTER_1;
      mask = REG1_MASK_HUMIDITY_TEMP_RESOLUTION;
      value = REG1_VALUE_RESOLUTION_H12_T14;
      _humidityDelay = DELAY_HUMIDITY12BIT_TEMP14BIT;
  }
  else if (setting == Settings::RESOLUTION_H08_T12) {
      writeRegister = CMD_WRITE_REGISTER_1;
      mask = REG1_MASK_HUMIDITY_TEMP_RESOLUTION;
      value = REG1_VALUE_RESOLUTION_H08_T12;
      _humidityDelay = DELAY_HUMIDITY8BIT_TEMP12BIT;
  }
  else if (setting == Settings::RESOLUTION_H10_T13) {
      writeRegister = CMD_WRITE_REGISTER_1;
      mask = REG1_MASK_HUMIDITY_TEMP_RESOLUTION;
      value = REG1_VALUE_RESOLUTION_H10_T13;
      _humidityDelay = DELAY_HUMIDITY10BIT_TEMP13BIT;
  }
  else if (setting == Settings::RESOLUTION_H11_T11) {
      writeRegister = CMD_WRITE_REGISTER_1;
      mask = REG1_MASK_HUMIDITY_TEMP_RESOLUTION;
      value = REG1_VALUE_RESOLUTION_H11_T11;
      _humidityDelay = DELAY_HUMIDITY11BIT_TEMP11BIT;
  }
  else if (setting == Settings::ADC_FAST) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_ADC_SPEED;
      value = REG2_VALUE_ADC_FAST;
      _adcDelay = DELAY_ADC_FAST;
  }
  else if (setting == Settings::ADC_NORMAL) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_ADC_SPEED;
      value = REG2_VALUE_ADC_NORMAL;
      _adcDelay = DELAY_ADC_NORMAL;
  }
  else if (setting == Settings::VIN_BUFFERED) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_VIN_BUF;
      value = REG2_VALUE_VIN_BUFFERED;
  }
  else if (setting == Settings::VIN_UNBUFFERED) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_VIN_BUF;
      value = REG2_VALUE_VIN_UNBUFFERED;
  }
  else if (setting == Settings::VOUT_VDDD) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_VOUT;
      value = REG2_VALUE_VOUT_VDDD;
  }
  else if (setting == Settings::VOUT_GNDD) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_VOUT;
      value = REG2_VALUE_VOUT_GNDD;
  }
  else if (setting == Settings::VREFP_VDDA) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_VREFP;
      value = REG2_VALUE_VREFP_VDDA;
  }
  else if (setting == Settings::VREFP_125V) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_VREFP;
      value = REG2_VALUE_VREFP_125V;
  }
  else if (setting == Settings::ADC_NO_HOLD) {
      writeRegister = CMD_WRITE_REGISTER_2;
      mask = REG2_MASK_ADC_HOLD;
      value = REG2_VALUE_ADC_NO_HOLD;
  }
  else if (setting == Settings::ADC_HOLD) {
	  writeRegister = CMD_WRITE_REGISTER_2;
	  mask = REG2_MASK_ADC_HOLD;
	  value = REG2_VALUE_ADC_HOLD;
  }
  else if (setting == Settings::ENABLE_HEATER) {
	  writeRegister = CMD_WRITE_REGISTER_1;
	  mask = REG1_MASK_ENABLE_HEATER;
	  value = REG1_VALUE_ENABLE_HEATER;
  }
  else if (setting == Settings::DISABLE_HEATER) {
	  writeRegister = CMD_WRITE_REGISTER_1;
	  mask = REG1_MASK_ENABLE_HEATER;
	  value = REG1_VALUE_DISABLE_HEATER;
  }
  else if (setting == Settings::ENABLE_THERMISTOR_CORRECTION) {
	  writeRegister = CMD_WRITE_REGISTER_2;
	  mask = REG2_MASK_ENABLE_THERMISTOR_CORRECTION;
	  value = REG2_VALUE_ENABLE_THERMISTOR_CORRECTION;
  }
  else if (setting == Settings::DISABLE_THERMISTOR_CORRECTION) {
	  writeRegister = CMD_WRITE_REGISTER_2;
	  mask = REG2_MASK_ENABLE_THERMISTOR_CORRECTION;
	  value = REG2_VALUE_DISABLE_THERMISTOR_CORRECTION;
  } else {
      return false;
  }
  
  return writeRegister8(writeRegister, value, mask);
}

bool Si7013::startMeasurement(uint8_t cmd) {
#ifdef DEBUG
Serial.print("startMeasurement()");
Serial.print(", status: ");
Serial.print(_status);
Serial.print(", cmd: ");
Serial.println(cmd);
#endif // DEBUG
   
  if (getStatus() != STATUS_IDLE) return false;

  if (cmd == CMD_MEASURE_HUMIDITY_HOLD || cmd == CMD_MEASURE_HUMIDITY_NO_HOLD ) {
    if (_adcNoHold) {
      while (changeSetting(Settings::ADC_HOLD) == false);
	  _adcNoHold = true;	// Reset the change made in changeSetting
    }
    _status = STATUS_MEASURING_HUMIDITY;
    _measurementDelay = ((float)_humidityDelay) * _delayMultiplier;
#ifdef DEBUG 
Serial.println("STATUS_MEASURING_HUMIDITY");
#endif    
  } 
  else if (cmd == CMD_MEASURE_TEMPERATURE_HOLD || cmd == CMD_MEASURE_TEMPERATURE_NO_HOLD ) {
    if (_adcNoHold) {
      while (changeSetting(Settings::ADC_HOLD) == false);
	  _adcNoHold = true;	// Reset the change made in changeSetting
    }
    _status = STATUS_MEASURING_TEMPERATURE;
    _measurementDelay = ((float)_temperatureDelay) * _delayMultiplier;
  } 
  else if (cmd == CMD_MEASURE_ADC) {
    if (_adcNoHold) {
      while (changeSetting(Settings::ADC_NO_HOLD) == false);
    }
    _status = STATUS_MEASURING_ADC;
    _measurementDelay = ((float)_adcDelay) * _delayMultiplier;
  } 

	_i2cPort->beginTransmission(_address);
	_i2cPort->write(cmd);
	_i2cPort->endTransmission();
	_measureStartTime = millis();
  
  return true;
}

bool Si7013::sendCommand(uint8_t cmd) {
#ifdef DEBUG
	Serial.print("sendCommand(): ");
	Serial.println(cmd);
#endif // DEBUG

	if (getStatus() != STATUS_IDLE) return false;

	_i2cPort->beginTransmission(_address);
	_i2cPort->write(cmd);
	if (_i2cPort->endTransmission())
		return false;
	
	return true;
}

bool Si7013::startHumidityTempMeasurement() {
#ifdef DEBUG
	Serial.println("startHumidityTempMeasurement()");
#endif // DEBUG

  return startMeasurement(CMD_MEASURE_HUMIDITY_NO_HOLD);
}


bool Si7013::startTempMeasurement() {
#ifdef DEBUG
	Serial.println("startTempMeasurement()");
#endif // DEBUG

	return startMeasurement(CMD_MEASURE_TEMPERATURE_NO_HOLD);
}

bool Si7013::startAdcMeasurement(){
#ifdef DEBUG
	Serial.println("startAdcMeasurement()");
#endif // DEBUG

  return startMeasurement(CMD_MEASURE_ADC);
}

float Si7013::getHumidity() {
#ifdef DEBUG
	Serial.println("getHumidity()");
#endif // DEBUG
	if (getStatus() == STATUS_IDLE) {
		while (_i2cPort->available() > 0) _i2cPort->read();	// discard any leftover data on the bus
		_i2cPort->requestFrom((uint8_t)_address, (uint8_t)3);
		uint32_t start = millis(); // start timeout
		while (millis() - start < _transactionTimeout) {
			if (_i2cPort->available() == 3) {
				uint16_t hum = _i2cPort->read() << 8 | _i2cPort->read();
				uint8_t chxsum = _i2cPort->read();
				// ToDo: use chxsum

				float humidity = hum;
				humidity *= 125;
				humidity /= 65536;
				humidity -= 6;

				_humidityNew = false;

				return humidity;
			}
		}
	}
	return NAN; // Error timeout
}

float Si7013::getPreviousTemperature() {
#ifdef DEBUG
	Serial.println("getPreviousTemperature()");
#endif // DEBUG
	if (getStatus() == STATUS_IDLE) {
		_i2cPort->beginTransmission(_address);
		_i2cPort->write(CMD_READ_PREVIOUS_TEMPERATURE);
		uint8_t err = _i2cPort->endTransmission();

		return getTemperature();
	}
	return NAN; // Error timeout
}

float Si7013::getTemperature() {
#ifdef DEBUG
	Serial.println("getTemperature()");
#endif // DEBUG
	if (getStatus() == STATUS_IDLE) {
		while (_i2cPort->available() > 0) _i2cPort->read();	// discard any leftover data on the bus
		_i2cPort->requestFrom((uint8_t)_address, (uint8_t)2);
		uint32_t start = millis(); // start timeout
		while (millis() - start < _transactionTimeout) {
			if (_i2cPort->available() == 2) {
				uint16_t temp = _i2cPort->read() << 8 | _i2cPort->read();

				float temperature = temp;
				temperature *= 175.72;
				temperature /= 65536;
				temperature -= 46.85;

				_temperatureNew = false;

				return temperature;
			}
		}
	}
	return NAN; // Error timeout
}

float Si7013::getAdc() {
#ifdef DEBUG
	Serial.println("getAdc()");
#endif // DEBUG
	if (getStatus() == STATUS_IDLE) {

		while (_i2cPort->available() > 0) _i2cPort->read();	// discard any leftover data on the bus

		uint32_t start = millis(); // start timeout
		_i2cPort->requestFrom((uint8_t)_address, (uint8_t)2);
		while (millis() - start < _transactionTimeout) {
			if (_i2cPort->available() == 2) {
				uint16_t analog = _i2cPort->read() << 8 | _i2cPort->read();

				_adcNew = false;

				return (float)analog;
			}
		}
	}
  return NAN; // Error timeout  
}

void Si7013::setTransactionTimeout(uint16_t transactionTimeout) {
	_transactionTimeout = transactionTimeout;
}

void Si7013::setMeasurementDelayMultiplier(float delayMultiplier) {
	_delayMultiplier = delayMultiplier;
}

/*!
 *  @brief  Reads serial number and stores It in sernum_a and sernum_b variable
 */
void Si7013::readSerialNumber() {
	_i2cPort->beginTransmission(_address);
	_i2cPort->write((uint8_t)(SI7021_ID1_CMD >> 8));
	_i2cPort->write((uint8_t)(SI7021_ID1_CMD & 0xFF));
	_i2cPort->endTransmission();

	bool gotData = false;
	uint32_t start = millis(); // start timeout
	while (millis() - start < _transactionTimeout) {
		if (_i2cPort->requestFrom((uint8_t)_address, (uint8_t)8) == 8) {
			gotData = true;
			break;
		}
		delay(2);
	}
	if (!gotData)
		return; // error timeout

	sernum_a = _i2cPort->read();
	_i2cPort->read();
	sernum_a <<= 8;
	sernum_a |= _i2cPort->read();
	_i2cPort->read();
	sernum_a <<= 8;
	sernum_a |= _i2cPort->read();
	_i2cPort->read();
	sernum_a <<= 8;
	sernum_a |= _i2cPort->read();
	_i2cPort->read();

	_i2cPort->beginTransmission(_address);
	_i2cPort->write((uint8_t)(SI7021_ID2_CMD >> 8));
	_i2cPort->write((uint8_t)(SI7021_ID2_CMD & 0xFF));
	_i2cPort->endTransmission();

	gotData = false;
	start = millis(); // start timeout
	while (millis() - start < _transactionTimeout) {
		if (_i2cPort->requestFrom((uint8_t)_address, (uint8_t)8) == 8) {
			gotData = true;
			break;
		}
		delay(2);
	}
	if (!gotData)
		return; // error timeout

	sernum_b = _i2cPort->read();
	_i2cPort->read();
	sernum_b <<= 8;
	sernum_b |= _i2cPort->read();
	_i2cPort->read();
	sernum_b <<= 8;
	sernum_b |= _i2cPort->read();
	_i2cPort->read();
	sernum_b <<= 8;
	sernum_b |= _i2cPort->read();
	_i2cPort->read();

	switch (sernum_b >> 24) {
	case 0:
	case 0xff:
		_model = SI_Engineering_Samples;
		break;
	case 0x0D:
		_model = SI_7013;
		break;
	case 0x14:
		_model = SI_7020;
		break;
	case 0x15:
		_model = SI_7021;
		break;
	default:
		_model = SI_UNKNOWN;
	}
}