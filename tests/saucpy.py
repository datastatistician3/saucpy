from pandas import read_csv
from saucpy import sAUC
#import saucpy

def main():
    p1 = sAUC.calculate_auc([2,.3,.4,2,1.2], [0.2, 1,2,1,2,3])
    print(p1)
    
if __name__ == '__main__':
    main()