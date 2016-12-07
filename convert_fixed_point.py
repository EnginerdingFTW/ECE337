import os


bit_precision = 12

def ConvertToFixedPoint(dir_path):
    read = open(dir_path, 'r')
    write = open(dir_path[0:len(dir_path) - 4] + "_fixed.txt", 'w')

    lines = read.readlines()

    temp = ""

    for line in lines:
        temp = temp + Convert(line) + "\n"

    write.write(temp)

    read.close()
    write.close()


def Convert(old):
    # print("original = " + old[:len(old)-1])
    value = ""

    if not ("." in old):
        value = ToBinary(old)
    else:
        fl = old.split('.')
        value = ToBinary(fl[0]) + "." + GetBinaryFraction(fl[1])

    # print("converted = " + value)
    return value

def GetBinaryFraction(val):
    val = float("0." + val)
    num = ""
    k = 1
    while (k <= bit_precision):
        if (val > (GetFloating(num) + (2 ** -k))):
            num = num + "1"
        else:
            num = num + "0"
        k = k + 1

    return num

def GetFloating(val):
    if (val == ""):
        return 0
    k = 1
    num = 0
    while (k <= len(val)):
        if (val[k-1] == "1"):
            num = num + 2**-k
        k = k + 1
    return num

def ToBinary(val):
    sign = "0"
    if (val[0] == "-"):
        sign = "1"
        val = val[1:]

    val = int(val)
    num = ""
    if (val == 0):
        num = "0"
    while (val > 0):
        if (val % 2 == 0):
            num = "0" + num
        else:
            num = "1" + num
        val = val // 2
    while (len(num) < 3):
        num = "0" + num
    num = sign + num
    return num

def BinToDec(st):
    num = 0
    k = 0
    sign_bit = 1
    if (st[len(st) - 1] == '\n'):
        st = st[:(len(st) - 1)]
    for letter in st:
        val = 0
        if (letter != "."):
            num = num * 2

        if (letter == "1"):
            val = 1
        if (k != 0):
            num = num + val
        else:
            if (val == 1):
                sign_bit = -1
        k = k + 1
        # print("letter = " + str(letter) + "  num = " + str(num))
    return num * sign_bit



def ConvertToDecimal(file):
    read = open(dir_path, 'r')
    write = open(dir_path[0:len(dir_path) - 9] + "dec.txt", 'w')

    lines = read.readlines()

    temp = ""
    for line in lines:
        num = 0
        if ("." in line):
            #is fixed point
            parts = line.split('.')
            t = GetFloating(parts[1])
            if (line[0] == "1"):
                t = t * -1
            num = BinToDec(parts[0]) + t
        else:
            #is regular binary
            num = BinToDec(line)
        temp = temp + str(num) + "\n"

    write.write(temp)

if __name__ == '__main__':
    file = raw_input("Enter the file name of the file to convert relative to this directory: ")
    choice = input ("To fixed - 1     or     To Decimal - 0    (1/0): ")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path + "/" + file
    if (choice == 1):
        ConvertToFixedPoint(dir_path)
    else:
        ConvertToDecimal(dir_path)