import sys

class Learnhmm():
    def __init__(self):
        self.dict = {}
        self.tag = {}
        self.data = []
        self.prior = []
        self.trans = []
        self.emit = []

    def get_dict(self,dict_file):
        with open(dict_file,'r') as f:
            # print(f.read())
            d = f.read().split('\n')
            while '' in d:
                d.remove('')
        self.dict = dict(zip(d, list(range(len(d)))))

    def get_tag(self,tag_file):
        with open(tag_file,'r') as f:
            # print(f.read())
            t = f.read().split('\n')
            while '' in t:
                t.remove('')
        self.tag = dict(zip(t, list(range(len(t)))))
        # print(self.tag)
    def get_data(self,train_in):
        with open(train_in,'r') as f:
            # print(f.read())
            sents = f.read().split('\n')
            while '' in sents:
                sents.remove('')
            for i in range(len(sents)):
                sent = sents[i].split()
                new_sent = []
                for j in range(len(sent)):
                    w_b = sent[j].split('_')
                    new_sent.append(w_b)
                self.data.append(new_sent)

    def get_data_wise(self,train_in,l):
        with open(train_in,'r') as f:
            # print(f.read())
            sents = f.read().split('\n')
            while '' in sents:
                sents.remove('')
            for i in range(len(sents)):
                if i==l:
                    break
                sent = sents[i].split()
                new_sent = []
                for j in range(len(sent)):
                    w_b = sent[j].split('_')
                    new_sent.append(w_b)
                self.data.append(new_sent)

    def learn(self):
        tag_d = dict(zip(self.tag.keys(),[0]*len(self.tag)))
        word_d = list([0]*len(self.dict))
        emit_d = {}
        trans_d = {}
        for k in self.tag:
            emit_d[k] = word_d.copy()
            trans_d[k] = tag_d.copy()

        for sent in self.data:
            # prior count & add
            tag_d[sent[0][1]] += 1
            self.prior = [(i+1)/(len(self.data)+len(tag_d)) for i in list(tag_d.values())]

            for j in range(len(sent)):
                # emit count
                tag = sent[j][1]
                word = sent[j][0]
                emit_d[tag][self.dict[word]] += 1

                # trans count
                if j ==0:
                    continue
                else:
                    yt = sent[j][1]
                    yt_1 = sent[j-1][1]
                    trans_d[yt_1][yt] += 1

        for k in self.tag:
            # emit add
            se = sum(list(emit_d[k]))
            le = len(emit_d[k])
            emit_mat = [(i+1)/(se+le) for i in emit_d[k]]
            self.emit.append(emit_mat)
            # trans add
            st = sum(list(trans_d[k].values()))
            lt = len(trans_d[k])
            trans_mat = [(i+1)/(st+lt) for i in trans_d[k].values()]
            self.trans.append(trans_mat)

        # print(self.prior)
        # print(self.trans)
        # print(self.emit)

    def write_result(self,prior_file,trans_file,emit_file):
        with open(prior_file,'w') as f:
            for i in self.prior:
                f.write('{:.18e}\n'.format(i))

        with open(trans_file,'w') as f:
            for tag in self.trans:
                s = ''
                for i in tag:
                    s += '{:.18e} '.format(i,'e')
                f.write('{}\n'.format(s))

        with open(emit_file,'w') as f:
            for tag in self.emit:
                s = ''
                for i in tag:
                    s += '{:.18e} '.format(i,'e')
                f.write('{}\n'.format(s))


if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word= sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    lhmm = Learnhmm()
    lhmm.get_dict(index_to_word)
    lhmm.get_tag(index_to_tag)
    lhmm.get_data(train_input)
    lhmm.learn()
    lhmm.write_result(hmmprior,hmmtrans,hmmemit)
    # print(lhmm.prior)
    # print(lhmm.dict)
    # print(lhmm.tag)