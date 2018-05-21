cnt =0
word = 0

with open('../rcv/test.feat','r') as fd:
    lines = fd.read().splitlines()
    for l in lines:
        w = l.split(" ")[1:]

        for x in w:
            word += int(x.split(":")[1])
        cnt += 1.
print(word/cnt)