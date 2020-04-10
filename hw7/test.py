import forwardbackward
import learnhmm
import matplotlib.pyplot as plt
import pickle
import numpy as np
if __name__ == '__main__':
    lhmm = learnhmm.Learnhmm()

    lhmm.get_dict('handout/index_to_word.txt')
    lhmm.get_tag('handout/index_to_tag.txt')
    # fw.get_d('handout/index_to_tag.txt','handout/index_to_word.txt')

    # lhmm.get_data_wise('handout/trainwords.txt', 10)
    # lhmm.learn()
    # lhmm.write_result('result/10prior.txt','result/10trans.txt','result/10emit.txt')
    # fw = forwardbackward.ForwardBackward()
    # fw.get_d('handout/index_to_tag.txt', 'handout/index_to_word.txt')
    # fw.get_model('result/10prior.txt','result/10trans.txt','result/10emit.txt')
    # fw.get_data('handout/trainwords.txt')
    # train10 = fw.pred_acc_log('result/10pred-train.txt', 'result/10met-train.txt')
    # fw.get_data('handout/testwords.txt')
    # test10 = fw.pred_acc_log('result/10pred-test.txt', 'result/10met-test.txt')
    # print(train10,test10)

    # lhmm.get_data_wise('handout/trainwords.txt', 100)
    # lhmm.learn()
    # lhmm.write_result('result/100prior.txt','result/100trans.txt','result/100emit.txt')
    # fw = forwardbackward.ForwardBackward()
    # fw.get_d('handout/index_to_tag.txt', 'handout/index_to_word.txt')
    # fw.get_model('result/100prior.txt','result/100trans.txt','result/100emit.txt')
    # fw.get_data('handout/trainwords.txt')
    # train100 = fw.pred_acc_log('result/100pred-train.txt', 'result/100met-train.txt')
    # fw.get_data('handout/testwords.txt')
    # test100 = fw.pred_acc_log('result/100pred-test.txt', 'result/100met-test.txt')
    # print(train100, test100)

    # lhmm.get_data_wise('handout/trainwords.txt', 1000)
    # lhmm.learn()
    # lhmm.write_result('result/1000prior.txt','result/1000trans.txt','result/1000emit.txt')
    # fw = forwardbackward.ForwardBackward()
    # fw.get_d('handout/index_to_tag.txt', 'handout/index_to_word.txt')
    # fw.get_model('result/1000prior.txt','result/1000trans.txt','result/1000emit.txt')
    # fw.get_data('handout/trainwords.txt')
    # train1000 = fw.pred_acc_log('result/1000pred-train.txt', 'result/1000met-train.txt')
    # fw.get_data('handout/testwords.txt')
    # test1000 = fw.pred_acc_log('result/1000pred-test.txt', 'result/1000met-test.txt')
    # print(train1000, test1000)

    # lhmm.get_data_wise('handout/trainwords.txt', 10000)
    # lhmm.learn()
    # lhmm.write_result('result/10000prior.txt','result/10000trans.txt','result/10000emit.txt')
    # fw = forwardbackward.ForwardBackward()
    # fw.get_d('handout/index_to_tag.txt', 'handout/index_to_word.txt')
    # fw.get_model('result/10000prior.txt','result/10000trans.txt','result/10000emit.txt')
    # fw.get_data('handout/trainwords.txt')
    # train10000 = fw.pred_acc_log('result/10000pred-train.txt', 'result/10000met-train.txt')
    # fw.get_data('handout/testwords.txt')
    # test10000 = fw.pred_acc_log('result/10000pred-test.txt', 'result/10000met-test.txt')
    # print(train10000, test10000)
    x = np.log([10,100,1000,10000])
    train = [-122.54156723373207,-110.31599954262037,-101.07201479899953,-95.43736971788367]
    test = [ -124.25345621103381,-111.40158980139648,-101.868242791792,-96.01832892261073]
    with open('./plot_lr.pkl','wb') as f:
        pickle.dump([x,train,test],f)
    plt.plot(x,train,ls='--', marker='o', ms = 1.5,lw=1,label='train avg log-likelihood')
    plt.plot(x,test,ls='-.', marker='v', ms = 1.5,lw=1,label='test avg log-likelihood')
    plt.xlabel('log(# Sequences)')
    plt.ylabel('avg log-likelihood')
    plt.title('Foward-Backward avg log-likelihood')
    plt.legend()
    plt.savefig('./1.4.png')
    plt.show()