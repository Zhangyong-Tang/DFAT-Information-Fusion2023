
def dada(i):
    for x in range(10):
        try:
            if x == 5:
                return(x)
                break
        except:
            print(100)


if __name__ == '__main__':
    aa = dada(3)
    print(aa)

