#!/bin/bash

import configparser
import optparse
import subprocess as cmd
import os
import sys
import re
import pdb
import queue

global LASTSPACENUM
LASTSPACENUM = queue.LifoQueue(maxsize = 20)
LASTSPACENUM.put(0)
LASTSPACENUM.put(0)


global CURPATH
global PARPATH
global THISSPACENUM
#global NEXTSPACENUM
THISSPACENUM = 0
#NEXTSPACENUM = 0

def run(cmd):# run a command
    status, output = cmd.Popen(cmd)
    return status, output

def initPath(): # init curpath and parpath
    global CURPATH, PARPATH
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    PARPATH = os.path.dirname(CURPATH)
    sys.path.append(PARPATH)
    sys.path.append(CURPATH)
    print("Path: ",CURPATH, PARPATH)

def readFile(filename):
    f = open(filename)
    cont = f.read()
    return cont

def lineCnt(line):
    baseline = line
    clrLine = re.sub("^ *","",line)
    cntSpace = len(re.findall("^ *",line))
    return baseline, clrLine, cntSpace

def getLastSpace():
    global LASTSPACENUM
    return LASTSPACENUM.get()

def putThisSpace(cnt):
    global LASTSPACENUM
    LASTSPACENUM.put(cnt)

def headWith(inp):
    global LASTSPACENUM
    global THISSPACENUM
    #if len(re.findall("^ *",line))>THISSPACENUM:
    #    THISSPACENUM = len(re.findall("^ *",line))
    #    return baseline
    baseLine, line, cntSpace = lineCnt(inp)
    line+="      "
    clrLine = line
    if False:#cntSpace < THISSPACENUM-5:
        if line[:-5] == "with ":
            resLine = " " * (THISSPACENUM-4) + clrLine
            return resLine
        elif line[:-4].find("def "):
            resLine = " " * (THISSPACENUM-4) + clrLine
            return resLine
        elif line[:-4].find("for "):
            resLine = " " * (THISSPACENUM-4) + clrLine
            return resLine
        elif line[:-3].find("if "):
            resLine = " " * (THISSPACENUM-4) + clrLine
            return resLine
    if line[:6] == "class ":
        putThisSpace(THISSPACENUM)
        putThisSpace(THISSPACENUM+4)
        resLine = " " * THISSPACENUM +  clrLine
        return resLine
    elif line[:3] == "if " or  line[:4] == "try ":
        _ = getLastSpace()
        resLine = " " * (_) + clrLine
        putThisSpace(_)
        THISSPACENUM=_+4
        return resLine
    elif line[:4] == "for ":
        _ = getLastSpace()
        resLine = " " * (_) + clrLine
        putThisSpace(_)
        putThisSpace(THISSPACENUM+4)
        THISSPACENUM=_+4
    elif line[:4] == 'def ' or line[:4] == "for " or line[:5]== "with ":
        _ = getLastSpace()
        resLine = " " * (_) + clrLine
        putThisSpace(_)
        THISSPACENUM=_+4
        return resLine
    elif line[:4] in ["else", "elif"] or line[:6] in ['except',"except"]:
        _ = getLastSpace()
        putThisSpace(_)
        resLine = " " * _ + clrLine
        THISSPACENUM += 4
        return resLine
    elif line[-1] == "/":
        resLine = " " * THISSPACENUM +  clrLine
        THISSPACENUM += 4
        return resLine
    else:
        resLine = ""
        #_ = getLastSpace()
        #if _-cntSpace>8:
        #    resLine = " " * _+ clrLine
        #elif _-cntSpace>4:
        #    resLine = " " * _+ clrLine
        #elif _-cntSpace>0:
        #    resLine = " " * _+ clrLine
        #else:
        #    resLine = " " * THISSPACENUM+ clrLine
        #putThisSpace(_)
        resLine = " " * THISSPACENUM+ clrLine
        return resLine

def formatLine(line):
    #line = re.sub("^ *","",line)
    line = re.sub("    ","    ",line)
    if line == "":
        return line
    line = headWith(line)
    return line

def buildPath(path):
    _ = os.path.join(CURPATH, path)
    return  _

def formatCode(readFileName, removeNote=True):
    _readFile = buildPath(readFileName)
    _writeFile= buildPath(readFileName +".format")
    #cmd = "touch %s"% writeFile
    #print("cmd: ",cmd)
    #output = cmd.Popen(cmd)
    #print(output)
    write_file = open(_writeFile, "a+")
    cont = readFile(_readFile)
    if removeNote:
        cont = re.sub("^ *$\n","",cont)
        cont = re.sub("\"\"\"[\d\D]+?\"\"\"","",cont)
    #print(cont)
    #pdb.set_trace()
    lines = cont.split('\n')
    removeNote = False
    for line in lines:
        # pdb.set_trace()
        print(line)
        line = formatLine(line)
        if line == "" or line == None:
            continue
        print(line)
        write_file.write(line+"\n")
        print(line)
    print(readFileName, "is now write into",readFileName,".format")

def getConfig(ini):
    try:
        cfg = configparser.ConfigParser()
        cfg.readfp(open(ini))
        print(cfg.sections())
    except:
        pass

if __name__=='__main__':
    parser = optparse.OptionParser()

    parser.add_option(
        "-i",
        "--ini",
        dest="ini",
        default="config.ini",
        help="read config from INI file",
        metavar="INI"
        )
    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        help="write report to FILE",
        metavar="FILE"
        )
    parser.add_option(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
        default=True,
        help="don't print status messages to stdout"
        )
    parser.add_option(
        "-r",
        "--login",
        dest="login",
        action="call_back",
        dest=""
        default=True,
        help="log in some server"
    (options, args) = parser.parse_args()
    getConfig(options.ini)
    print(args)

