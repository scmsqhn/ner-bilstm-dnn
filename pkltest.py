#ValueError: invalid literal for int() with base 10: '问'
import pickle
with open('/home/siyuan/data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    #get_ipython().magic('time X = pickle.load(inp)')
    #get_ipython().magic('time y = pickle.load(inp)')
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
print('** Finished loading the data.')    
print(word2id)
print(word2id[['张','海','昌','民']])
#print(id2word)
#print(tag2id)
#print(id2tag)
print(X[0])
#print(y[0])

#============================================================================================================
