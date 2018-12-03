import random

newdata = []
with open("./data/combine.csv")as f:
    for each in f.readlines():
        linedata = []
        count = 1
        line = each.strip().split(",")
        label = int(line[0])-1
        if label>=3:
            continue
        linedata.append(str(label))
        for each2 in line[1:]:
            if count%37!=1:
                # print(each2)
                linedata.append(each2)
            count+=1
        newdata.append(linedata)
# print(newdata)
# split train and test
test_rate = 0.08
random.shuffle(newdata)
test_num = int(len(newdata)*test_rate)

train_num = len(newdata)-test_num

training_data = newdata[:train_num]
test_data = newdata[train_num:]
print(test_num,train_num)
# generate training data
with open('./data/RP/RP_train.csv', 'w') as f:
    for each in training_data:
        for each2 in each:
            f.write(each2+",")
        f.write("\n")
# generate testdata
with open('./data/RP/RP_test.csv', 'w') as f:
    for each in test_data:
        for each2 in each:
            f.write(each2+",")
        f.write("\n")
        