from django.apps import AppConfig
from deepface.basemodels import ArcFace
import faiss
import json
import numpy as np


class ArcfaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'arcface'
    all_embed = np.load('arcface/arc_face.npy')
    embedding_model = ArcFace.loadModel()
    searcher = faiss.IndexFlatL2(512)
    with open('id_dict.txt') as f:
        data = f.read()
    id_dict = json.loads(str(data))

    with open('all_id_dict.txt') as d:
        all_data = d.read()
    all_id_dict = json.loads(str(all_data))

