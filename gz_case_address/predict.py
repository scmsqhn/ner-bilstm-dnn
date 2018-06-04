import tensorflow as tf
import sys
sys.path.append("..")
sys.path.append(".")
from .model import BiLSTM_CRF
import os, argparse, time, random
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding
from dutil.pycrypt import *
from dutil.utility import get_config
import re

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

args = {
    'batch_size': 128,
    'epoch': 20,
    'hidden_dim': 300,
    'optimizer': 'Adam',
    'CRF': True,
    'lr': 0.001,
    'clip': 5.0,
    'dropout': 0.8,
    'update_embedding': True,
    'shuffle': True
}

## get char embeddings
#word2id = read_dictionary(os.path.join(os.environ['DMPPATH'],'gz_case_address/data_path/word2id.pkl'))
word2id = read_dictionary("./gz_case_address/data_path/word2id.pkl")
embeddings = random_embedding(word2id, 300)

## paths setting
#output_path = os.path.join(os.environ['DMPPATH'],'dmp/gongan/gz_case_address/mode_save')
output_path = os.path.join("./gz_case_address/mode_save")
# output_path = ('./mode_save')

if not os.path.exists(output_path):
    os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints")
if not os.path.exists(model_path):
    os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
result_path = os.path.join(output_path, "results")
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
get_logger(log_path).info(str(args))

g = tf.Graph()
ckpt_file = tf.train.latest_checkpoint(model_path)


model = BiLSTM_CRF(batch_size=args["batch_size"], epoch_num=args["epoch"], hidden_dim=args["hidden_dim"],
                   embeddings=embeddings,
                   dropout_keep=args["dropout"], optimizer=args["optimizer"], lr=args["lr"], clip_grad=args["clip"],
                   tag2label=tag2label, vocab=word2id, shuffle=args["shuffle"],
                   model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,graph=g,
                   CRF=args["CRF"], update_embedding=args["update_embedding"])


# training model
def train():
    train_path = os.path.join('./data_path/', 'train_data')
    test_path = os.path.join('./data_path/', 'test_data')

    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path);
    test_size = len(test_data)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    model.train(train_data, test_data)  # we could use test_data as the dev_data to see the overfitting phenomena

model.build_graph()
# init the netwDork
with g.as_default():
    saver = tf.train.Saver()
sess = tf.Session(graph=g)
saver.restore(sess, ckpt_file)

# ONE sentence predict
def ner_predict(cases,encrypt=False):
    result = []
    for case_str in cases:
        pc = prpcrypt(get_config().get("encryptKey", "key"))
        #process the string or listinfo
        if encrypt:
            case_str = pc.decrypt(bytes(case_str, encoding='utf-8'))
            case_str = re.sub("\|","",case_str)
            case_str = re.sub("    ", "", case_str)
            case_str = re.sub("\n", "", case_str)
            case_str = re.sub(" ", "", case_str)
        predict_sent = list(case_str.strip())
        demo_data = [(predict_sent, ['O'] * len(predict_sent))]
        tag = model.demo_one(sess, demo_data)
        LOC = get_entity(tag, predict_sent)
        res = {
            "loc":list(set(LOC)),
        }
        result.append(res["loc"])
    return result

if __name__ == '__main__':
    from collections import Counter
    # train()
    # wp,non_WP,wp_c = batch_predict()
    # wp = Counter(wp)
    # print(wp.keys())
    # print(wp.values())
    # print(non_WP)   # print(wp_c)
    #
    # sentence = "报3月27号晚上在十八里店环岛东玉盛大酒店门口，出租车司机给我掉包了150元假币，出租车车牌号：京BQ0056。"
    # sentence = "柳江县基隆开发区八号地，我停放在这里的一辆小车不知道被什么车撞坏了。"
    # sentence = "王先生报警称：环城高速往桂林方向柳北服务区，5点多我驾驶一辆货车（鲁G26739）到这里加油，离开时我忘记取走加油卡了，卡内还有1万多元钱，我现在已经离开服务区100多公里了，因为当时是加油站的工作人员帮我输的密码，我虽然挂失了油卡，但挂失生效需要1个小时，我担心这段时间内油卡会被人使用，所以想查询一下服务区的电话。"
    # sentence = "柳江河中间（靠文庙这边），我在这里钓鱼看到一个箱子浮在水中间，我觉得这个箱子很可疑，现在箱子还在水里面，我不敢捞起来看是什么。"
    # sentence = "高新二路沿江路口军事博物馆对面非机动车道上，我路过看见有很多小车违章停放在这里，影响通行了"
    sentence = "04:39分（13558225383）荣军路宝山宾馆，我们是伤者的家属，我们现在工人医院陪护他，之前我亲戚在那里发生了交通事故，我有肇事司机的联系电话，我们现在想知道我们能怎么处理？"
    # sentence = "报警人称逮住一偷电瓶的小偷。。民警到达现场后已将该人员带到派出所进一步处理。"
    # print(ner_predict([sentence]), sentence)
    '''
    import codecs
    with codecs.open("./data_pre/data/gz_add_val.txt",'r',encoding="utf-8") as fr:
        with codecs.open("./data_pre/data/gz_add_res.txt",'wb',encoding="utf-8") as fw:
            for sentence in fr.readlines():
                # sentence = "经出警核实，系报警人何可建（男，身份证号码：520103199110020414）从贵阳市云岩区君子巷57号2单元附10号家中起床时，发现家中被入室盗窃，经清点核实，其放在卧室床边左边的一部步步高手机、3000元人民币现金及放在客厅桌子上的一部惠普笔记本电脑被盗，经现场初步侦查，系为技术性开锁入室盗窃，财物信息：现金3000元人民币；手机（牌子：步步高，型号：X6，串号不详，颜色：银白色，购于：2015年12月，购价：2500元人民币，手机号：18798086394）；电脑（牌子：惠普，型号不详，购于：2014年1月，购价：4500元人民币），我所报立刑事案件侦查。"
                # sentence = "未反馈报警人称被技术性开锁入室，家里未被翻乱，就是其爱人的钱包被盗，内有1000余元现金"
                # sentence = "工作中发现，2017年11月09日上午，我局民警在芝罘区胜利路如家酒店检查中发现徐飞（男，43"
                if len(sentence.strip())>0:
                    print(ner_predict([sentence.strip()]),sentence)
                    # print(sentence)
                    # fw.write(str(ner_predict(sentence)['loc'])+"　"+sentence)
    '''
