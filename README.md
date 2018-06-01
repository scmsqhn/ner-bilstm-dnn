# 使用 bilstm和dnn进行命名实体提取

# 测试结果 使用5000样本进行地址提取训练，准确率90+



# 要点
１　使用专业词库在训练前进行分词,比较此前在模型后使用地址词库处理，效果有提升
２　６０００（有效样本约３０００）贵州警察情训练数据在太原的预测值约６０％－７０％　毛估估
３　增加样本和专业词库，再一次训练．
４  逻辑部分正确较高
５　３０００抽样２７３未能检出　错误在１０％-２０％

# 错误分析
４　未找出的中None占绝大部分，即模型从未录入该ｄｉｃｔＩＤ
５　//太原本地名词，同上
６　相对方位词，如房间内，外面，后面

# 基本可以初步可以判断，词库＋神经网路可以覆盖所有的ＮＥＲ任务


目的是将介词动词发现出来
分词的词典尽可能长且完整
同时,词典需要使用id 词向量会因为训练文本的偏置导致,神经网路模型的收敛更难

256
9 梯度 越大越容易跳过局部最有解 也容易不收敛 收敛慢
学习率使用adm固定 1e-4
使用自己优化动态


DNN RNN dnn 
