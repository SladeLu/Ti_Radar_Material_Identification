import serial
import time
import struct
import matplotlib.pyplot as plt

CONFIG_FILEPATH = 'profile_2d_DroneWater.cfg'

CLICOM = "COM6"
CLIBAUDURATE = 115200

DATACOM = "COM5"
DATABAUDURATE = 921600

DATASET_SIZE = 3000

def InitialCLI(CONFIG_FILEPATH,COM,BAUDURATE):
    config = []
    with open(CONFIG_FILEPATH, 'r') as f:
        for eachline in f.readlines():
            config.append(eachline)

    port = serial.Serial(COM,BAUDURATE,timeout=1)
    if port.isOpen():
        print("open success")
    else:
        print("open failed")
        return
    for each in config:
        data = bytes(each,'ascii')
        port.write(data)
        time.sleep(0.1)
    port.close()
    print("Initialize CLI SUCCESS")

def DataProcess(DATA):
    #magic 0201040306050807
    data = DATA[16:]
    rangeProfileTemp = data[96:]

    data_float =[]
    index = 0
    while index<len(data)-8:
        data_float.append(data[index:index+8])
        index +=8

    rangeProfile_sint16 = []
    index = 0
    while index<len(rangeProfileTemp)-4:
        temp = struct.unpack('H',bytes.fromhex(rangeProfileTemp[index:index+4]))[0]
        temp = float(temp)
        if temp > 2**15:
            temp = temp-2**16
        rangeProfile_sint16.append(temp)
        index +=4

    index = 0
    rp_cplx = []
    while index<len(rangeProfile_sint16)-2:
        rp_cplx.append(rangeProfile_sint16[index]+rangeProfile_sint16[index+1]*1j)
        index += 2  

    globalCountTemp = data_float[0]
    outPhase = struct.unpack('f',bytes.fromhex(data_float[3]))[0]
    outConfidenceMetric = struct.unpack('f',bytes.fromhex(data_float[4]))[0]
    outEnergyWfm = struct.unpack('f',bytes.fromhex(data_float[5]))[0]
    outRcsVal = struct.unpack('f',bytes.fromhex(data_float[6]))[0]
    outWeightedSum = struct.unpack('f',bytes.fromhex(data_float[7]))[0]

    maxRangeBinIndexTemp = data[88:96]
    outMaxRangeBinIndex = struct.unpack('H',bytes.fromhex(maxRangeBinIndexTemp[4:]))[0]

    # print("recieve",outWeightedSum)
    return outRcsVal,outWeightedSum

# InitialCLI(CONFIG_FILEPATH,CLICOM,CLIBAUDURATE)
rcs = []
weightsum = []
num = 0
with serial.Serial(DATACOM,DATABAUDURATE,timeout=30) as dataport:
    while True:
        count = dataport.inWaiting()
        if count > 0:
            data = dataport.read(count)
            r,w = DataProcess(data.hex())
            rcs.append(r)
            weightsum.append(w)
            if num>=DATASET_SIZE:
                break
            else :
                num += 1
        
plt.figure()
plt.plot(rcs)
plt.show()




