import warnings

class MA(object):

    count = 0

class SimpleMA(MA):

    def __init__(self, n, empty=0):
        self.n = int(n)
        self.empty = empty
        self.lasts = []
        

    def compute(self, value):
        self.count+=1
        self.lasts.append(float(value))
        length = len(self.lasts)

        #Slice lasts if it contains more than n values
        if length > self.n:
            self.lasts = self.lasts[-self.n:]
            length = self.n

        # Calculate MA value of the last n value in list. Uses as much values as disponible.
        MA = (sum(self.lasts))/length
        # Test if we have exactly n value or more and pop the first to be ready for next round
        # if not enough check emty to return the right thing
        if length == self.n:
            self.lasts.pop(0)
            return MA
        elif not self.empty :
            # empty have a false value provided
            return MA
        else:
            return self.empty

class EMA(MA):

    def __init__(self, a):
        self.a = a
        self.last = 0

    def compute(self, value):
        #data is list of ordered value wich is already clean and numerical
        if  self.count == 0 :
            self.last = float(value)
        else:
            self.last = self.a *float(value) + (1-self.a)*float(self.last)
        
        self.count = self.count+1
        return self.last

class NDayEMA(EMA):

    def __init__(self, n):
        self.n = n
        if n < 2:
            warnings.warn("ATTENTION: N should probably be bigger thant 2.")
        try:
            a = 2.0000/(self.n + 1)
            super(NDayEMA, self).__init__(a) # init the parent class with a
        except ZeroDivisionError:
            raise Execption("ERROR: N should not be equal to 1, ZeroDivisionError.")
            exit()

class CMA(MA):
    def __init__(self):
        self.last = 0

    def compute(self, value):
        self.count+=1
        self.last+=float(value)
        return self.last / self.count

class WMA(MA):
    def __init__(self, n, empty=0):
        self.n = n
        self.empty = empty
        self.lasts = []
        
    def compute(self, value):
        self.lasts.append(float(value))
        length = len(self.lasts)

        #Slice lasts if it contains more than n values
        if length > self.n:
            self.lasts = self.lasts[-self.n:]
            length = self.n


        self.weight = range(1, length + 1)
        self.weight.reverse()
        denominator = sum(self.weight)
        # Calculate MA value of the last n value in list. Uses as much values as disponible.
        WMA = sum([last*self.weight.pop() for last in self.lasts])/denominator
        # Test if we have exactly n value or more and pop the first to be ready for next round
        # if not enough check emty to return the right thing
        if length == self.n:
            self.lasts.pop(0)
            return WMA
        elif not self.empty :
            # empty have a false value provided
            return WMA
        else:
            return self.empty


class EMA20(NDayEMA):

    def __init__(self):
        super(EMA20, self).__init__(20)

class EMAW(NDayEMA):

    def __init__(self):
        super(EMA20, self).__init__(7)

class EMA7(NDayEMA):

    def __init__(self):
        super(EMA20, self).__init__(7)

class EMA5(NDayEMA):

    def __init__(self):
        super(EMA20, self).__init__(5)
