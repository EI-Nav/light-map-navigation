import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src/llm_obj_nav/llm_obj_nav/instructnav"))
from cv_utils.object_list import categories

GLEE_CONFIG_PATH = "/home/lab417/llm-osm/light-map-navigation/src/llm_obj_nav/llm_obj_nav/instructnav/thirdparty/GLEE/configs/SwinL.yaml"
GLEE_CHECKPOINT_PATH = "/home/lab417/llm-osm/light-map-navigation/src/llm_obj_nav/llm_obj_nav/instructnav/thirdparty/GLEE/GLEE_SwinL_Scaleup10m.pth"
DETECT_OBJECTS = [[cat['name'].lower()] for cat in categories]
INTEREST_OBJECTS = ['bed','chair','toilet','potted_plant','television_set','sofa']


