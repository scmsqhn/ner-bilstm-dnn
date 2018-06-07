#
import os 
CUR_PATH = os.path.join(os.environ['DMPPATH'], 'iba/dmp/gongan/storm_crim_classify/extcode')
import sys
sys.path.append(CUR_PATH)
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
