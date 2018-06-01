
white  = ['z','b','i','e','s']
phoneNum = ['p','h','n']
wx = ['u','v','x']
identifier = ['d','t','f']
qq = ['Q','q', 'k']
creditCard = ['c','r','R']
carinfo = ['y','w','Y']
mail = ['m','N','M']
web = ['o','P','O']
nickname = ['W', 'E', 'C']
momo = ['1','2','3']
#qqname = ['4','5','6']
#wxname = ['7','8','9']

tags = []
tags.extend(white)
tags.extend(phoneNum)
tags.extend(wx)
tags.extend(identifier)
tags.extend(qq)
tags.extend(creditCard)
tags.extend(carinfo)
tags.extend(mail)
tags.extend(web)
tags.extend(nickname)
tags.extend(momo)
#tags.extend(qqname)
#tags.extend(wxname)

assert len(tags) == len(list(set(tags)))

key_name_lst = ["white","phoneNum","wx","identifier","qq","creditCard","carinfo","mail","web","nickname","momo"]#,"qqname"," wxname"]
key_value_lst = [white,phoneNum,wx,identifier,qq,creditCard,carinfo,mail,web,nickname,momo]#,qqname, wxname]

assert len(key_name_lst) == len(key_value_lst)

key_dict = dict(zip(key_name_lst, key_value_lst))

ll = []
for i in range(len(tags)):
    ll.append(i)

tags_dict = dict(zip(tags, ll))
num_dict = dict(zip(ll, tags))

