column=[(1,2),(1,3),(1,4),(2,4),(2,5),(3,4),(3,5),(4,5)]
value = [0,0,0,0,0,0,0,0]
with open('ok.txt') as f:
    for line in f:
        strt = line
        strt2 = strt.split(':')
        my_result = tuple(map(int, strt2[0].split(',')))
        for i in range(len(column)):
        	if(column[i]==my_result):
        		value[i] = int(strt2[1])
print(value)
