
#include "EmotiBit_NCP5623.h"

NCP5623::NCP5623() {}
/*
For setting the I2C port and dvice address of the chip.
It also sets the LED pwm values and Iout current value
*/
bool NCP5623::begin(TwoWire &wirePort) {
  _i2cPort = &wirePort;
  _deviceAddress = NCP5623_DEFAULT_ADDR;

  // Checking if sensor exists on the I2C line
  _i2cPort->beginTransmission(_deviceAddress);
	_i2cPort->write(0x00);
	if (_i2cPort->endTransmission())
		return false;
	// Set the Vout Current to a value before using PWM on the led's
    setCurrent(30);
	// Set the LED to a maximum 
	setLEDpwm(1, 31);
	setLEDpwm(2, 31);
	setLEDpwm(3, 31);
	setLED(1, false);
	setLED(2, false);
	setLED(3, false);

	return true;
}

/*
Function returns the state of the LED at the requested position
*/
bool NCP5623::getLED(uint8_t ledPosition) {
	return _stateLed[ledPosition - 1];
}

/*
Function to set an a LED on or off
*/
void NCP5623::setLED(uint8_t ledPosition, bool state) {
	if (_stateLed[ledPosition - 1] != state) {
		_ledChanged[ledPosition - 1] = true;
	}
	_stateLed[ledPosition - 1] = state;
}

// send LED state & PWM value to LED controller chip
void NCP5623::send() {
  uint8_t reg;
	for (uint8_t i = 0; i < NCP_5623_NUM_LED; i++){
		if (_ledChanged[i]) {
			reg = NCP5623_REG_CHANNEL_BASE + i;
			uint8_t val;
			if (_stateLed[i]) { // Switch ON led based on set pwm level 
				val = ((reg & 0x7) << 5) | (getLEDpwm(i + 1) & 0x1F);
			}
			else { // switch OFF led
				val = (reg & 0x7) << 5;
			}
			_i2cPort->beginTransmission(_deviceAddress);
			_i2cPort->write(val);
			_i2cPort->endTransmission();
		}
	}
}



/*
Function returns the pwm value for a LED position.
*/
uint8_t NCP5623::getLEDpwm(uint8_t ledPosition) {

	return _pwmValLed[ledPosition - 1];
}

/*
Function sets the PWM value for a led position
*/
void NCP5623::setLEDpwm(uint8_t ledPosition, uint8_t pwm_val) {
	pwm_val = (pwm_val > 31) ? 31 : pwm_val;
	pwm_val = (pwm_val < 0) ? 0 : pwm_val;
	if (_pwmValLed[ledPosition - 1] != pwm_val)
	{
		_ledChanged[ledPosition - 1] = true;
	}
	_pwmValLed[ledPosition - 1] = pwm_val;
}

/*
Function sets the MAX output current.
*/
void NCP5623::setCurrent(uint8_t iled) {
	// Todo: make current setting asyncronous
    iled = (iled>30)?30:iled;
    _i2cPort->beginTransmission(_deviceAddress);
    _i2cPort->write(((NCP5623_REG_ILED&0x7)<<5)|(iled&0x1f)); // rrrvvvvv
    _i2cPort->endTransmission();
}

/*
Fucntion to print statements to the console for debugging purposes.
*/
void NCP5623::enableDebugging(Stream &debugPort)
{
	_debugPort = &debugPort;
	_printDebug = true;
}