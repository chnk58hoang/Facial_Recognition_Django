import random
import pandas as pd
from django.core.files.images import ImageFile
from django.core.files.storage import FileSystemStorage
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.views import APIView
from rest_framework.response import Response
from tqdm import tqdm
from .apps import ArcfaceConfig
import time
import os
import numpy as np
from deepface import DeepFace
from .models import Person


class Search(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'images/search_face.html'

    def search(self, img_url):
        base_dir = 'arcface/upload'
        ArcfaceConfig.searcher.add(ArcfaceConfig.all_embed)
        img_path = os.path.join(base_dir, img_url)
        start = time.time()
        face = DeepFace.detectFace(img_path=img_path, target_size=(112, 112), detector_backend='mtcnn', align=True)
        face = face[:, :, (2, 1, 0)]
        detect_time = time.time()
        face = np.expand_dims(face, axis=0)
        face_embeded = ArcfaceConfig.embedding_model.predict(face)[0:, ]
        extract_time = time.time()
        dist, ids = ArcfaceConfig.searcher.search(face_embeded, k=1)
        search_time = time.time()
        print(ArcfaceConfig.all_id_dict[img_url])
        return ArcfaceConfig.id_dict[str(ids[0][0])], ArcfaceConfig.all_id_dict[img_url], round(
            detect_time - start, ndigits=3), round(
            extract_time - detect_time, ndigits=3), round(search_time - extract_time, ndigits=3)

    def get(self, request):
        return Response(None)

    def post(self, request):
        fs = FileSystemStorage(location='arcface/upload')
        files = request.FILES.getlist('file')
        try:
            filename = fs.save(files[0].name, files[0])
            if "_" in filename:
                filename = filename.split("_")[0] + '.jpg'
            img_path = '/arcface/upload/' + filename
            try:
                id_num, true_id, detect_time, extract_time, search_time = self.search(filename)
                img = Person.objects.get(identity_number=id_num).image

                print(img_path)
                print(img.url)
                # fs.delete(filename)
                return Response(data={'id': id_num, 'true_id': true_id, 'img': img, 'upload_img': img_path,
                                      'detect': str(detect_time),
                                      'extract': str(extract_time), 'search': str(search_time)})
            except ValueError:
                return Response(data={'id': None, 'upload_img': img_path})
        except IndexError:
            return Response(None)


def create(request):
    df = pd.read_csv('id1.csv')
    base_dir = 'images'
    for i in tqdm(range(len(df))):
        image_name = df.iloc[i]['image']
        id = df.iloc[i]['id']
        image_path = os.path.join(base_dir, image_name)

        person = Person.objects.create(identity_number=id)
        person.image = ImageFile(open(image_path, "rb"))
        person.save()


class MAP(APIView):

    def search(self, img_name, k):
        img_path = os.path.join('images/', img_name)
        ArcfaceConfig.searcher.add(ArcfaceConfig.all_embed)
        try:
            face = DeepFace.detectFace(img_path=img_path, target_size=(112, 112), detector_backend='mtcnn', align=True)
            face_embeded = ArcfaceConfig.embedding_model.predict(np.expand_dims(face, axis=0))[0:, ]
            dist, ids = ArcfaceConfig.searcher.search(face_embeded, k=k)
            tp = 0
            ap = 0
            true_id = int(ArcfaceConfig.all_id_dict[img_name])
            for i in range(len(ids[0])):
                if ids[0][i] <= 10157:
                    id_num = ArcfaceConfig.id_dict[str(ids[0][i])]

                    if id_num == true_id:
                        tp += 1
                        ap += float(tp / (i + 1))
            return ap / tp
        except ValueError:
            print('lol')


    def get(self, request):
        map = 0.0
        for i in range(50):
            randomfile = random.choice(os.listdir('images'))
            if "_" in randomfile:
                randomfile = randomfile.split("_")[0] + '.jpg'

            map += self.search(randomfile, k=5)
        map = map / 50
        return Response(map)
