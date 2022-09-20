import os
import random

from django.core.files.images import ImageFile
from rest_framework.views import APIView
from tqdm import tqdm

from .models import Person
from .apps import RecognizerConfig
from .forms import AddForm
from deepface import DeepFace
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer
from django.core.files.storage import FileSystemStorage
import numpy as np
import time
import pandas as pd


# Create your views here.


class Search(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'images/search_face.html'

    def search(self, img_url):
        base_dir = 'facenet512/upload'
        RecognizerConfig.searcher.add(RecognizerConfig.all_embed)
        img_path = os.path.join(base_dir, img_url)
        start = time.time()
        target_img = DeepFace.detectFace(img_path=img_path, target_size=(160, 160), detector_backend='mtcnn',align=True)
        #target_img = target_img[:, :, (2, 1, 0)]
        detect_time = time.time()
        target_representation = RecognizerConfig.embedding_model.predict(np.expand_dims(target_img, axis=0))[0, :]
        target_representation = np.expand_dims(target_representation, axis=0)
        extract_time = time.time()
        dist, ids = RecognizerConfig.searcher.search(target_representation, k=1)
        search_time = time.time()
        return RecognizerConfig.id_dict[str(ids[0][0])], RecognizerConfig.all_id_dict[img_url], round(
            detect_time - start, ndigits=3), round(
            extract_time - detect_time, ndigits=3), round(search_time - extract_time, ndigits=3)

    def get(self, request):
        return Response(None)

    def post(self, request):
        fs = FileSystemStorage(location='facenet512/upload')
        files = request.FILES.getlist('file')
        try:
            filename = fs.save(files[0].name, files[0])
            if "_" in filename:
                filename = filename.split("_")[0] + '.jpg'
            img_path = '/facenet512/upload/' + filename
            print(img_path)
            try:
                id_num, true_id, detect_time, extract_time, search_time = self.search(filename)
                img = Person.objects.get(identity_number=id_num).image
                print(img.url)
                # fs.delete(filename)
                return Response(data={'id': id_num, 'true_id': true_id, 'img': img, 'upload_img': img_path,
                                      'detect': str(detect_time),
                                      'extract': str(extract_time), 'search': str(search_time)})
            except ValueError:
                return Response(data={'id': None, 'upload_img': img_path})
        except IndexError:
            return Response(None)


class AddFace(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'images/add_face.html'

    def get(self, request):
        add_form = AddForm()
        return Response({"form": add_form})

    def post(self, request):
        form = AddForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            id = form.cleaned_data['id']

            person = Person.objects.create(image=image, identity_number=id)
            person.save()


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
        RecognizerConfig.searcher.add(RecognizerConfig.all_embed)
        try:
            face = DeepFace.detectFace(img_path=img_path, target_size=(160, 160), detector_backend='mtcnn', align=True)
        except ValueError:
            print('lol')
        face_embeded = RecognizerConfig.embedding_model.predict(np.expand_dims(face, axis=0))[0:, ]
        dist, ids = RecognizerConfig.searcher.search(face_embeded, k=k)
        print(ids)
        tp = 0
        ap = 0
        true_id = int(RecognizerConfig.all_id_dict[img_name])
        for i in range(len(ids[0])):
            if ids[0][i] <= 10157:
                id_num = RecognizerConfig.id_dict[str(ids[0][i])]

                if id_num == true_id:
                    tp += 1
                    ap += float(tp / (i + 1))
        return ap / tp

    def get(self, request):
        map = 0.0
        for i in range(100):
            randomfile = random.choice(os.listdir('images'))
            if "_" in randomfile:
                randomfile = randomfile.split("_")[0] + '.jpg'

            map += self.search(randomfile, k=5)
        map = map / 100
        return Response(map)
