# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:41:09 2022

v0.1: derived from TScan. 
v0.2 changlog:
    -Added functions for 2 MSM. 
    -Renamed functions for better clarity.
    -trying to avoid return values in bytes.
    
@author: malte.henrich
"""

import serial
from serial.tools import list_ports_windows
from serial import SerialException
import time
import re

class TScan:
    
    def __init__(self):
        self._tscan = serial.Serial()
        self._tscan.baudrate = 921600
        self._tscan.port = None
        self._tscan.timeout = 20
        self.is_open = self._tscan.is_open   
        self.MSM = 'VIS'
        self.info = {'SN#':'',
                    'Version':'',
                    'Revision':'',
                    'Date':'',
                    'SN#-TScan':''}     

    def __del__(self):
        self.disconnect()
        

# ##########   ERROR HANDLING   ########## #

    def __test_for_error(self, status):
        if status == b'\x15':
            self.__get_error(-1)
        elif status == b'\x06'or status == b'\r' or status == 13 or status == 6:
            return True
        else:
            self.__get_error(status)
        pass
    
    def __get_error(self, code):
        if code == (-1, 21):
            raise NameError("Unknown Error in command.")
        elif code == -2:
            raise NameError("TSError: Wrong Parameter in command")
        elif type(code) == tuple:
            raise NameError(code[0])
        elif type(code) == str and code.startswith("TSError"):
            raise NameError(code)
        else:
            raise NameError(f"Unknown Error. {code}")
        pass
    
# ##########   INTERNAL METHODS   ########## #

    def _read_all_para(self): # special case of all para read. Ends with \x03 not with \x06 or \r
        try:
            self._tscan.write(bytes('*para:allpara?\r', 'utf-8'))
            buf_size = self._tscan.in_waiting
            response = self._tscan.read(buf_size)
            #if response[-1] != b'\x03':
            #    if self._tscan.in_waiting == 0:
            #        self.__get_error("TSError: Unknown Error.")
        except SerialException as ex:
            self.__get_error(ex.args)
        return response
                
    def _read_response(self, commandstr):
        try:
            self._tscan.write(bytes(commandstr + '\r', 'utf-8'))
        except SerialException as ex:
            self.__get_error(ex.args)
        response = list()
        try:
            while True:
                response.append(self._tscan.read(1))  # read byte by byte
                if response[-1] == b'\r': # read until end of line character or bell 
                    break
                elif response[-1] == b'\x06': # or read until ACK
                    break
                elif response[-1] == b'\x15':  # or read until NACK
                    break
            return b''.join(response)
        except SerialException as ex:
            self._tscan.close()
            self.__get_error(ex.args)
            #return -1
    
    def _get_info(self):
        self.info['SN#-TScan'] = self._read_response('*TSCAN:IDN?').decode().strip()
        self.info['Revision'] = self._read_response('*TSCAN:REV?').decode().strip()
        self.info['Version'] = self._read_response('*TSCAN:VERSION?').decode().strip()
        self.info['SN#'] = self._read_response('*TSCAN:SERN?').decode().strip()
        self.info['Date'] = self._read_response('*TSCAN:DATE?').decode().strip()
        
        
# ##########   CONNECT   ########## #

    def find_device(self):
        ''' Get Port of connected device. If more than one device is connected, 
            only first one found is returned.
        '''
        devlist = list_ports_windows.comports()
        print(devlist)
        for p in devlist:
            try:
                if p.pid == 19520:
                    if p.vid == 6421:
                        tmp = serial.Serial(p.device)
                        tmp.write(b'*TSCAN:VERSION?\r')
                        time.sleep(1)
                        resp = tmp.read(tmp.in_waiting).decode()
                        if len(resp) != 0:
                            tmp.close()
                            return p.device
            except UnicodeDecodeError:
                tmp.close()
                continue
            except SerialException as e:
                self.__get_error(e.args)
                print(f"Error {e.strerror()} while trying port {p.device} \n")  
                #print(e)
                pass
        return None
        
    def connect(self, portname=""):
        ''' Connect and Open Port to TScan device.
            params:
                portname: Name of Port, eg. 'COM10' [OPTIONAL]
            returns:
                boolean True if succesful.
        '''
        if not self.is_open:
            if portname == "":
                self._tscan.port = self.find_device()
            else:
                self._tscan.port = portname
            if self._tscan.is_open:
                self._tscan.close()
            self._tscan.open()
            self.is_open = self._tscan.is_open
            self._get_info()
        if self.is_open:
            return True
        else:
            return False
            
    def disconnect(self):
        ''' Disconnect TScan device from Port. '''
        if self._tscan.is_open:
            self._tscan.close()
            self.is_open = self._tscan.is_open
        if not self.is_open:
            return True
        else:
            return False
            
        
# ##########   METHODS   ########## #

    def LED_on(self, chn, amp, ontime=0):
        ''' Turn on LED channel 'chn'. 
            params:
                chn: channel to turn on. Can be either a number in [1,2,3,4] 
                     or string in ['325nm', '365nm', '390nm', '450nm']
                amp: supplied current in mA
                ontime: time until LED turns off automatic in ms [OPTIONAL, DEFAULT = 0]. 
                        If ontime is 0 LED stays on until expliccitly turn off.
        '''
        if type(amp) != int:
            amp = int(amp)
        if type(chn) != int:
            if type(chn) == str:
                if chn in ['325nm','365nm','390nm','450nm','OFF']:
                    led=chn
                else:
                    self.__get_error(-2)
        amp_max = 700
        if chn == 1:  
            led = '325nm'
            #amp_max = 350
            amp_max = 700
        elif chn == 2:
            led = '365nm'
            amp_max = 700
        elif chn == 3:
            led='390nm'
            #amp_max = 350
            amp_max = 700
        elif chn == 4:
            led='450nm'
            amp_max = 1000
        else:
            self.__get_error(-2)
        if amp > amp_max:
            self.__get_error(f"TSError: Current to high! Max. allowed current for LED: {amp_max} mA.")
        if ontime == 0:
            cmd = '*TSCAN:LED {} {}'.format(led, amp)
        else:
            cmd = '*TSCAN:LED {} {} {}'.format(led, amp, ontime)
        resp = self._read_response(cmd)
        self.__test_for_error(resp[-1])
        
    def LED_off(self):
        ''' Turn LED off. '''
        cmd = '*TSCAN:LED OFF 0 0'
        resp = self._read_response(cmd)
        self.__test_for_error(resp[-1])    
        
    def set_mode(self, m):
        ''' Set Firmware mode. Currently only 0 works, switches to normal mode.
            ATTENTION: If mode is set to 0 TScan will turn off and can not 
            be reached with this class anymore.
            params:
                m: number of mode.
        '''
        cmd = '*TSCAN:MODE {}'.format(m)
        resp = self._read_response(cmd)
        self.__test_for_error(resp[-1])  
        
    def get_temperatures(self):
        '''Get Temperature readings from TScan device.
            returns: list(float) of temperatures. First value is T of OFE head, 
                     second is T on mainboard near constant current supply 
        '''
        cmd = '*TSCAN:TEMP?'
        resp = self._read_response(cmd)
        resp_str = resp.decode()
        ts = resp_str.split('\t')
        ret = [float(ts[0]), float(ts[1].strip())]
        self.__test_for_error(resp[-1])
        return ret     
    
    def get_LED_current(self):
        ''' Get actual current for LED.
            returns: float of current in mA.
        '''
        cmd='*TSCAN:CURRENT?'
        resp = self._read_response(cmd)
        self.__test_for_error(resp[-1])
        ret = float(resp.decode())
        return ret
        
    def get_electronics_serial(self):
        ''' Get Serial Number of TScan Electronics.
            returns: string Serialnumber.'''
        return self.info['SN#']
        # cmd = '*TSCAN:SERN?'
        # return self._read_response(cmd)
        
    def get_firmware_version(self):
        ''' Get Firmware Version.
            returns: string Firmware Version.'''
        return self.info['Version']
        # cmd = '*TSCAN:VERSION?'
        # return self._read_response(cmd)
        
    def get_tscan_serial(self):
        ''' Get Serialnumber of TScan Device
            returns: string Serialnumber'''
        return self.info['SN#-TScan']
    
    def get_electronics_revision(self):
        ''' Get REvison of Electronics Board.
            returns: string of revions. '''
        return self.info['Revision']
    
    def get_firmware_date(self):
        ''' Get date of firmware build.
            returns: string date of build.'''
        return self.info['Date']
    
    def spec_measure(self, tint, avg):
        '''Condcut a MSM measurement.
            params:
                tint: integration time in ms
                avg: number of averages
            returns:
                measured counts per pixel as list of integers. 
        '''
        inBuffer = self._tscan.in_waiting
        if inBuffer != 0:
            self._tscan.read(inBuffer)
        #print('ts_measure start')
        spec = list()
        pixRange = self._read_response('*PARA:PIXRANGE?')  # request pixel range from device
        pixRange = list(map(int, re.findall(r'\d+', pixRange.decode('utf8'))))  # convert to int

        expectedBytes = pixRange[1] - pixRange[0] + 1  # calculate expected size of measurement

        self._tscan.write(bytes(f'*MEAS:DARK {tint} {avg} 1\r', 'utf8'))  # write measurement command
        #first_byte = self._tscan.read(1) # get rid of first byte (additional hex06)
        response = list() 
        while True:
            response.append(self._tscan.read(2))
            if len(response) >= expectedBytes + 2:  # because i already read first byte only one additional byte expected Expect 2 more bytes ACK and BEL
                #last_byte = self._tscan.read(1)
                break
        for i in response[2:]:
            spec.append(int.from_bytes(i, "little"))
        return spec          
    
    def get_wavelengths_params(self):
        ''' Get list of wavelength fit parameters.
        returns:
            a,b,c values as float.
        '''
        a =float(self._read_response('*PARA:FIT2?').decode())
        b =float(self._read_response('*PARA:FIT1?').decode())
        c =float(self._read_response('*PARA:FIT0?').decode())
        return a,b,c
    
    def set_wavelengths_params(self, a, b, c):
        ''' Set wavelength fit parameters
        params:
            a: float of second order parameter
            b: float of first order parameter
            c: float of zero order parameter
        '''
        self._read_response('*PARA:FIT2 {}'.format(a))
        self._read_response('*PARA:FIT1 {}'.format(b))
        self._read_response('*PARA:FIT0 {}'.format(c))
        self._read_response('*PARA:SAVE')
        
    def send_command(self, command):
        ''' Send a command to TScan.
            params:
                command: string of command
            returns:
                string depending on response of device.
        '''
        if command == '*para:allpara?':
            resp = self._read_all_para()
        resp = self._read_response(command)
        self.__test_for_error(resp[-1])
        return resp.decode()
    
    def send_command2(self, command):
        if command == '*MEAS:TEMPE':
            #resp = self._read_all_para()
            resp = self._read_response(command)
        return resp[:6].decode("utf-8")
    
    def get_msm_serial(self):
        ''' Get MSM serial number.
            returns:
            serialnumber as string.
        '''
        return self._read_response('*para:spnum?').decode()
    
    def measure(self, tint, avg, chn, amp):
        ''' Conduct a measurement with TScan LEDs.
            params:
                tint: integration time in ms
                avg: number of averages
                chn: LED channel to use. must be in [1,2,3,4]
            returns:
                measured counts per pixel as list of integers.
        '''
        if chn not in [1,2,3,4]:
            print("Channel not implemented.")
            return -1
        if chn == 1:
            #amp_max = 350
            amp_max = 700
        elif chn == 2:
            amp_max = 700
        elif chn == 3:
            #amp_max = 350
            amp_max = 700
        elif chn == 4:
            amp_max = 1000
        else:
            amp_max= 700

        if amp > amp_max:
            self.__test_for_error("TSError: Power to high!")
            
        self.LED_on(chn, amp)
        spec = self.spec_measure(tint, avg)
        self.LED_off()
        return spec
    
    def get_msm(self):
        ''' Get the current active MSM channel. 
            returns:
                'VIS' or 'NIR'
        '''
        cmd = '*TSCAN:SPEC?'
        resp = self._read_response(cmd)
        self.__test_for_error(resp[-1])
        ret = resp.decode()[:-1]
        return ret
        
    def set_msm(self, msm):
        ''' Set the active MSM channel.
            params:
                msm: channel to activate (possible values [0,1,'VIS','NIR'])
            returns:
                success.
        '''
        if type(msm) == int:
            if msm == 0:
                p ='VIS'
            elif msm == 1:
                p = 'NIR'
            else:
                self.__get_error(-2)
                print("wrong parameter (0, 1)")
        elif msm in ['VIS', 'NIR']:
            p = msm
        else:
            self.__get_error(-2)
            print("Wrong parameter ('VIS', 'NIR')")
        cmd = '*TSCAN:SPEC {}'.format(p)
        resp = self._read_response(cmd)
        if not self.__test_for_error(resp[-1]):
            return False
        
# ---------- Future Functions for multiple spectrometer TSCANs. ---------- # 
    
    

