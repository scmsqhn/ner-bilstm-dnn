# timer.py
import collections
import datetime

class TimeCnt():
    def __init__(self):
        self.deque = collections.deque(maxlen=3)
        self.milestone = list()
        self.marker = 0
        self.time_mark()

    def time_mark(self):
        _time = datetime.datetime.now()
        self.deque.append(_time)

    def cnt_time(self):
        self.time_mark()
        self.marker+=1
        print("> 节点:", self.marker ,"耗时:", (self.deque[-1]-self.deque[-2]).microseconds,"毫秒")



if __name__ == "__main__":
    _tc = TimeCnt()
    _tc.cnt_time()
    _tc.cnt_time()
    _tc.cnt_time()

