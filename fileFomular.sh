#!/bin/bash

trorev=$1

if [ $trorev -eq 0 ] # file into debug mode
  then 
    echo "change the file  into debug mode"
    sed 's/^[ ]*//g' bilstm_train.py eval_bilstm.py datahelper.py

    sed 's/^ *//g'  bilstm_train.py eval_bilstm.py datahelper.py
    sed 's/^[[:space:]]*//g'  bilstm_train.py eval_bilstm.py datahelper.py

    sed -i "s/DEBUG = False/DEBUG = True/g" bilstm_train.py eval_bilstm.py datahelper.py
elif [ $trorev -eq 1 ]
  then
    #echo $trorev
    echo "nohup python eval_bilstm.py > tmp 2>&1 &"
elif [ $trorev -eq 20 ]
  then
    #> output文本: 云岩区麻冲路335号飞讯网吧门前
    #> text: 到现场报警人陈城男52010319871106241213985522174称其位于延安中路109号2单元附14号被入室盗窃被盗一台三星A7手机价值2400元一台苹果5办业务送的两台酷派手机价值不详一台苹果平板电脑价值3800元立刑案 
    #> wordid_rever: 到现场报警人陈城男None称其位于延安中路109号2单元附14号被入室盗窃被盗一台三星A7手机价值2400元一台苹果5办业务送的两台None价值不详一台苹果None价值3800元立刑案 
    #> mark文本: 
    echo "clr the predict result to output"
    sed -e "s/>wordid.+$\n//g" ./predictable.txt
    sed -e "s/>mark.+$\n/\n==========\n/g" ./predictable.txt
    sed -e "s/^$\n//g" ./predictable.txt
else
  echo 'else'
  fi



#"""
#
#2、行后和行前添加新行
#	   行后：sed 's/pattern/&\n/g' filename
#
#3、使用变量替换(使用双引号)
#    sed -e "s/$var1/$var2/g" filename
#
#4、在第一行前插入文本
#    sed -i '1 i\插入字符串' filename
#
#5、在最后一行插入
#    sed -i '$ a\插入字符串' filename
#
#6、在匹配行前插入
#    sed -i '/pattern/ i "插入字符串"' filename
#
#7、在匹配行后插入
#   sed -i '/pattern/ a "插入字符串"' filename
#
#8、删除文本中空行和空格组成的行以及#号注释的行
#   grep -v ^# filename | sed /^[[:space:]]*$/d | sed /^$/d
##"""
