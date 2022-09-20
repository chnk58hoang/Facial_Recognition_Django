from django.apps import AppConfig
from deepface.basemodels import Facenet512
import numpy as np
import faiss
import json


class RecognizerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'facenet512'
    all_embed = np.load('facenet512/all_embedded.npy')
    embedding_model = Facenet512.loadModel()
    searcher = faiss.IndexFlatL2(512)

    with open('id_dict.txt') as f:
        data = f.read()
    id_dict = json.loads(str(data))

    with open('all_id_dict.txt') as d:
        all_data = d.read()
    all_id_dict = json.loads(str(all_data))
