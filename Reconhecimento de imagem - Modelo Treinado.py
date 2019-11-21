# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:22:09 2019
Neste script abaixo foi treinado um modelo da API do IBM Watson para
identificar as compatibilidades da imagem passada a ele
@author: ADAMLINCOLNOLIVEIRAS
"""
#______________________________________Importando as bibliotecas____________________________________________________________________________________#
import json
from os.path import abspath
from ibm_watson import VisualRecognitionV3, ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator   #Nesta versão mais nova é preciso importar o pacote de autenticação.

#______________________________________Parte de autenticação da API_________________________________________________________________________________#
authenticator = IAMAuthenticator('_ayFQ7pYNoN7uy0_MWooNipeHk1TWUmyeB0UBKaTmNRn') #API KEY Neste ponto a achave de autenticação esta sendo passada para o tipo Autenticação'
visual_recognition = VisualRecognitionV3( '2018-03-19', authenticator=authenticator) #Entre parenteses a API Key gerada pelo Watson Studio.
visual_recognition.set_service_url('https://gateway.watsonplatform.net/visual-recognition/api') #Setando o gatway da APi do watson. É encontrado junto da API keyKey.

#______________________________________Iniciando o treinamento do modelo____________________________________________________________________________#

#Passando repositorio de imagens para treinar o modelo, 3 positivos e 1 negativo.
with open('./cachorros/pug.zip', 'rb') as pug,  open('./cachorros/Palemao.zip', 'rb') as Palemao, open('./cachorros/goldenretriever.zip', 'rb') as goldenretriever, open('./cachorros/notdog.zip', 'rb') as notdog:
    model = visual_recognition.create_classifier('dogs', positive_examples={'pug': pug, 'Palemao': Palemao, 'goldenretriever': goldenretriever}, negative_examples=notdog).get_result()
print(json.dumps(model, indent=2))


#______________________________________Testando o modelo____________________________________________________________________________#


dog_path = abspath('./cachorros/ImagensDeTeste/4.jpg')  #Caminho da imagem de teste

try:
    with open(dog_path, 'rb') as images_file:
        dog_results = visual_recognition.classify(
            images_file=images_file,
            threshold='0.1',
            classifier_ids=['dogs_585908991']).get_result()   #classifier_ids serve para definir qual modelo irá chamar, o nome do modelo é exebido no json de treinamento.
        print(json.dumps(dog_results, indent=2))              #Existem 3 modelos padrão, Default, Food e Explicit.
except ApiException as ex:
    print(ex)





