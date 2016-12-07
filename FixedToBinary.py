import os

def ConvertToBinary(file):
    read = open(file, 'r')
    write = open(file[0:(len(file) - 9)] + "bin.txt", 'w')
    lines = read.readlines()

    string_ = ""
    for line in lines:
        num = 0
        temp = line[::-1]
        line1 = line[0:9]
        line2 = line[9:(len(line) - 1)]
        # print("line = " + line + "  line1 = " + line1 + "  line2 = " + line2)
        for letter in line1:
            val = 0
            if (letter != "."):
                num = num * 2

            if (letter == "1"):
                val = 1
            num = num + val
        string_ = string_ + chr(num)
        num1 = num


        num = 0
        for letter in line2:
            val = 0
            if (letter != "."):
                num = num * 2

            if (letter == "1"):
                val = 1
            num = num + val
        string_ = string_+ chr(num)
        num2 = num

        # print("num1 = " + str(num1) + "   num2 = " + str(num2))
    write.write(string_)




if __name__ == '__main__':
    file = raw_input("Enter the file name in relative to this directory: ")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path+ "/" + file
    print(dir_path)
    ConvertToBinary(dir_path)