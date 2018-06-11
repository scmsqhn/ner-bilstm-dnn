#　运行测试代码
./pred.sh

#　文件
数据接口　datahelper.py 
验证文件　eval_bilstm.py

#  接口
import ner_crim_addr_guiyang as ncag


from ancag.eval_bilstm import Eval_Ner
eval_ins = Eval_Ner()
result = eval_ins.predict_txt(['我在学校门口马路边抢到一分钱'])
result = eval_ins.predict_txt(['我在学校门口马路边偷到一分钱'])
result = eval_ins.predict_txt(['我在学校门口马路边骗到一分钱'])
result = eval_ins.predict_txt(['我在学校门口马路边卖毒品赚偷到一分钱'])



#　依赖包安装
pip install -r requirment.txt

