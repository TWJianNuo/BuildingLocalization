import random
class randomDNASeq:
    def __init__(self, seqLength, prefixSeq = ''):
        if seqLength < 10:
            raise ValueError('Sequence lenth too small')
        dnaSeqList = list()
        dnaChar = ['A', 'T', 'G', 'C']
        for i in range(seqLength):
            idx = random.randint(0,3)
            dnaSeqList.append(dnaChar[idx])
        self.dnaSeq = ''.join(dnaSeqList)

        if len(prefixSeq) > 0:
            self.dnaSeq = ''.join(prefixSeq)

        curLen = 0
        lenRange = [5, 20]
        self.dnaFrag = list()
        while(seqLength - curLen > 22):
            randSeqLen = random.randint(5, 20)
            self.dnaFrag.append(self.dnaSeq[curLen : curLen + randSeqLen])
            curLen = curLen + randSeqLen - 3
        self.dnaFrag.append(self.dnaSeq[curLen:])
        random.shuffle(self.dnaFrag)
    def printAllDnaFragments(self):
        for idx, frag in enumerate(self.dnaFrag):
            print("%dth Frag is:%s" % (idx, frag))
    def printEntireDna(self):
        print("Entire DNA seq is :%s" % self.dnaSeq)
    def getEntireDnaSeq(self):
        return self.dnaSeq
    def getDnaSeqments(self):
        return self.dnaFrag
    def checkCorrectness(self, results):
        if self.dnaSeq == ''.join(results):
            return 1
        else:
            return -1
class unitSeq:
    def __init__(self, dnaStr):
        self.start = dnaStr[0:3]
        self.end = dnaStr[-3:]
        self.dnaStr = dnaStr
    def merge(self, new_unitSeq):
        mergedList = list()
        if self.start == new_unitSeq.end:
            newStr = new_unitSeq.dnaStr + self.dnaStr[3:]
            mergedList.append(unitSeq(newStr))
        if self.end == new_unitSeq.start:
            newStr = self.dnaStr + new_unitSeq.dnaStr[3:]
            mergedList.append(unitSeq(newStr))
        return mergedList



def recurs(mergedSeq, inputSeq):
    if len(inputSeq) == 0:
        return
    newEle = inputSeq.pop()
    for idx, ele in enumerate(mergedSeq):
        mergeRe = ele.merge(newEle)
        if len(mergeRe) > 0:
            for re_ele in mergeRe:
                popped = mergedSeq.pop(idx)
                inputSeq.append(re_ele)
                recurs(mergedSeq, inputSeq)
                if len(inputSeq) == 0 and len(mergedSeq) == 1:
                    return
                else:
                    mergedSeq.insert(idx, popped)
                    inputSeq.pop()
    mergedSeq.append(newEle)
    recurs(mergedSeq, inputSeq)
    if len(inputSeq) == 0 and len(mergedSeq) == 1:
        return
    else:
        mergedSeq.pop()
        inputSeq.append(newEle)

"""
s0 = dnaSeq.dnaFrag[0]
s1 = dnaSeq.dnaFrag[1]
s2 = dnaSeq.dnaFrag[2]
s3 = dnaSeq.dnaFrag[3]

s0_unit = unitSeq(s0)
s1_unit = unitSeq(s1)
s2_unit = unitSeq(s2)
s3_unit = unitSeq(s3)


tSe1 = 'AAACBBB'
tSe2 = 'BBBCAAA'
tSe1_unit = unitSeq(tSe1)
tSe2_unit = unitSeq(tSe2)
m = tSe1_unit.merge(tSe2_unit)
print(m[0].dnaStr)
print(m[1].dnaStr)

c = s0_unit.merge(s1_unit)
print(c[0].dnaStr)

d = s2_unit.merge(s1_unit)
print(d[0].dnaStr)
"""
def checker(generated, seqs):
    generated_cpy = generated
    for i in range(5000):
        for seq in seqs:
            if generated[0:len(seq)] == seq:
                generated = generated[len(seq)-3 :]
                if len(generated) == 3:
                    print("Success")
                    return
    print("Fail")

for i in range(500):
    isSuccess = False
    seqLength = 100
    dnaSeq = randomDNASeq(seqLength)
    seqList = list()
    for frag in dnaSeq.getDnaSeqments():
        seqList.append(unitSeq(frag))

    mergedSeq = list()
    recurs(mergedSeq, seqList)
    checker(mergedSeq[0].dnaStr, dnaSeq.getDnaSeqments())
    """
    if len(mergedSeq) == 1 and dnaSeq.checkCorrectness(mergedSeq[0].dnaStr) == 1:
        isSuccess = True
    else:
        isSuccess = False

    if ~isSuccess:
        a = 1
    """

"""
for i in range(10):
    predefinedSeq = [
        'AAATTTBBB',
        'BBBTTTBBB',
        'BBBTTTCCC'
    ]
    random.shuffle(predefinedSeq)
    predefinedEntireSeq = 'AAATTTBBBTTTBBBTTTCCC'
    mergedSeq = list()
    seqList = list()
    for frag in predefinedSeq:
        seqList.append(unitSeq(frag))
    recurs(mergedSeq, seqList)
    if len(mergedSeq) == 1:
        print(mergedSeq[0].dnaStr == predefinedEntireSeq)
"""