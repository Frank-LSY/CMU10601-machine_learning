import numpy as np
import sys
from learnhmm import Learnhmm

class ForwardBackward():
    def __init__(self):
        self.lhmm = Learnhmm()
        self.dict = {}
        self.tag = {}
        self.prior = []
        self.trans = []
        self.emit = []
        self.data = []
        return

    def get_d(self,index_tag,index_word):
        self.lhmm.get_tag(index_tag)
        self.lhmm.get_dict(index_word)
        self.tag = dict(zip(self.lhmm.tag,range(len(self.lhmm.tag))))
        self.dict = self.lhmm.dict

    def get_model(self,hmmprior,hmmtrans,hmmemit):
        with open(hmmprior,'r') as f:
            self.prior = list(map(float,f.read().split('\n')[:-1]))
        with open(hmmtrans,'r') as f:
            lines = f.readline()
            while lines:
                # print(map(float,lines.split()))
                self.trans.append(list(map(float,lines.split())))
                lines = f.readline()
        with open(hmmemit,'r') as f:
            lines = f.readline()
            while lines:
                self.emit.append(list(map(float,lines.split())))
                lines = f.readline()

        self.prior = np.array(self.prior)
        self.trans = np.array(self.trans)
        self.emit = np.array(self.emit)

    def get_data(self,test_in):
        self.lhmm.get_data(test_in)
        self.data = self.lhmm.data

    def forward(self,i,sent):
        if i==0:
            # print(self.prior,self.emit[:,0])
            log_alpha_1_k = np.log(self.prior)+np.log(self.emit[:,self.dict[sent[0][0]]])
            return log_alpha_1_k
        else:
            log_bjxt = np.log(self.emit[:,self.dict[sent[i][0]]])
            log_akj = np.log(self.trans)
            log_alpha_t1_k = self.forward(i-1,sent)
            log_alpha_t1_a = log_akj.T+log_alpha_t1_k
            max_a = log_alpha_t1_a.max(axis=1)
            # print(max_a)
            #exp_log_alpha_t1_a = np.sum(np.exp(log_alpha_t1_a),axis=1)
            log_alpha_i = log_bjxt+max_a+np.log(np.exp(log_alpha_t1_a.T-max_a).sum(axis=0))
            return log_alpha_i

    def backward(self,i,sent):
        if i==len(sent)-1:
            log_beta_T = np.array([0]*len(self.prior))
            return log_beta_T
        else:
            bjxt1 = self.emit[:,self.dict[sent[i+1][0]]]
            log_bjxt1 = np.log(bjxt1)
            log_bjk = np.log(self.trans)
            # print(log_bjk)
            log_beta_t1_k = self.backward(i+1,sent)
            log_b_beta_t1_a = log_bjk+log_bjxt1+log_beta_t1_k
            max_b = log_b_beta_t1_a.max(axis=1)
            # exp_log_b_beta_t1_a = np.sum(np.exp(log_b_beta_t1_a),axis=1)
            log_beta_i = max_b+np.log(np.exp(log_b_beta_t1_a.T-max_b).sum(axis=0))
            return log_beta_i

    def forwardbackward(self,sent):
        forward = []
        backward= []
        # print(sent)
        for i in range(len(sent)):
            forward.append(self.forward(i,sent))
            backward.append(self.backward(i, sent))
        alpha = np.array(forward)
        beta = np.array(backward)
        log_p_yt = alpha+beta
        # # print(log_p_yt)
        alpha = np.exp(alpha)
        beta = np.exp(beta)
        p_yt = np.exp(log_p_yt)
        # print(np.sum(p_yt,axis=1))
        print(alpha)
        print(beta)
        return alpha, beta, log_p_yt

    def prediction(self,p_yt,sent):
        pred = []
        for i,p in enumerate(p_yt):
            tag_index = list(p).index(max(p))
            pred.append([sent[i][0],list(self.tag.keys())[tag_index]])
            # tag.append(tag_index)

        return pred

    def pred_acc_log(self,pred_out,met_out):
        # average log likelihood
        avg_log_arr = []
        # acc_arr = []
        acc = 0
        count = 0
        with open(pred_out,'w') as f:
            # prediction
            for sent in self.data:
                alpha,beta,p_yt = self.forwardbackward(sent)
                pred_sent= self.prediction(p_yt,sent)
                s_out = ''
                # print(sent)
                # print(pred_sent)
                for i in range(len(sent)):
                    s_out += '{}_{} '.format(pred_sent[i][0],pred_sent[i][1])
                    # accuracy
                    pre_tag = pred_sent[i][1]
                    ori_tag = sent[i][1]
                    if pre_tag == ori_tag:
                        acc += 1
                    count += 1
                f.write('{}\n'.format(s_out.rstrip()))

                # avg log likelihood
                max_alpha = np.max(alpha[-1])
                sum_log_alpha = np.sum(np.exp(alpha[-1]-max_alpha))
                avg_log_arr.append(max_alpha+np.log(sum_log_alpha))
                avg_log = sum(avg_log_arr) / len(avg_log_arr)
                # print(avg_log)

        print(acc/count)
        print(avg_log)
        with open(met_out,'w') as f:
            f.write('Average Log-Likelihood: {}\n'.format(avg_log))
            f.write('Accuracy: {}\n'.format(acc/count))

        return avg_log

if __name__ == '__main__':
    test_input = sys.argv[1]
    index_to_word= sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    fw = ForwardBackward()

    fw.get_d(index_to_tag,index_to_word)
    fw.get_model(hmmprior,hmmtrans,hmmemit)
    fw.get_data(test_input)
    # print(fw.emit)
    fw.pred_acc_log(predicted_file, metric_file)

