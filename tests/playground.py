import numpy as np
from datetime import datetime
import time
data = np.random.normal(0,1,(1000,1000))

timeStart = datetime.now()
print(f"test starts at {timeStart} ...")
for i in range(100):
    tmp = np.linalg.inv(data)
print(f"test ends with {datetime.now()-timeStart} ...")