# Fiap Computer Vision

## Introdução

Este repositório contém o código e os recursos para projetos de visão computacional desenvolvidos como parte do currículo da FIAP (Faculdade de Informática e Administração Paulista). O foco está em implementar diversas técnicas e algoritmos de visão computacional utilizando Python e bibliotecas populares como OpenCV e TensorFlow.

## Datasets Utilizados

Bases de Dados Utilizadas

- [LCC FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd)
- [Real vs Fake Anti-Spoofing Video Classification](https://www.kaggle.com/datasets/trainingdatapro/real-vs-fake-anti-spoofing-video-classification)
- [Anti-Spoofing](https://www.kaggle.com/datasets/tapakah68/anti-spoofing)

## Método de Geração de Dados

Foi realizado uma redução de amostras fakes, do dataset [LCC FASD]https://www.kaggle.com/datasets/faber24/lcc-fasd, e utilizado os vídeos reais dos datasets [Real vs Fake Anti-Spoofing Video Classification]https://www.kaggle.com/datasets/trainingdatapro/real-vs-fake-anti-spoofing-video-classification e [Anti-Spoofing]https://www.kaggle.com/datasets/tapakah68/anti-spoofing para a geração de novos frames, utilizando o `ffmpeg`e o script abaixo:

`
!ffmpeg -i caminho_do_video/{i}.mp4 -vf "select=gt(scene\,0.01)"  -vsync vfr caminho_do_saida_imagem/person_{i}_%04d.png
`

Em seguida foi utilizado a função `crop_faces` (`haarcascade`) para recortar apenas os rostos das pessoas da foto:


    
    def crop_faces(input_folder, output_folder):

    # Utiliza-se modelo pre-treinado de detecção de faces (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Se o caminho de saída não existir, ele é criado
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Utiliza-se um loop para pecorrer todas as imagens do caminho de entrada
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_folder, filename) # imagem
            output_image_path = os.path.join(output_folder, 'face_' + filename ) #face_imagem

            # Utiliza-se o OpenCV para ler as imagens
            image = cv2.imread(input_image_path)

            # Erro - Não foi possivel ler a imagem
            if image is None:
                print(f"Erro: Não foi possível ler a imagem {input_image_path}")
                continue

            # Converte-se a imagem para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Utiliza-se o haar cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            # Verifica-se se foi detectado alguma face
                # Caso nenhuma face seja detectada, retorna-se o warming.
            if len(faces) == 0:
                print(f"Warming: Nenhuma face detectada na imagem {input_image_path}")
                continue

                # Caso a imagem seja detectada, ela é recortada.
            for (x, y, w, h) in faces:
                cropped_face = image[y:y+h, x:x+w]
                break  # Crop only the first detected face

            # A face recortada é salva no caminho selecionado
            cv2.imwrite(output_image_path, cropped_face)
            print(f"Face salva em: {output_image_path}")
            

## Elaboração da solução

### Document Check

Foi utilizado a biblioteca face_recognition do Python, e criado uma função que utilizar que recebe a foto do documento e a do usuário, e realiza o checagem.


    def  document_check(foto_documento, foto_usuario):
        import cv2
        import face_recognition as fr
        imgDocumento = fr.load_image_file(foto_documento)
        imgDocumento = cv2.cvtColor(imgDocumento,cv2.COLOR_BGR2RGB)
        imgTest = fr.load_image_file(foto_usuario)
        imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
        imgDocumento_encode = fr.face_encodings(imgDocumento)[0]
        imgTest_encode = fr.face_encodings(imgTest)[0]
        comparacao = fr.compare_faces([imgDocumento_encode],imgTest_encode)
        
        if comparacao[0] == True:
            print("\nVerificação bem-sucedida | Documento confere com o usuário\n")
            plt.imshow(cv2.cvtColor(imgDocumento,cv2.COLOR_BGR2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("\nVerificação mal-sucedida | Documento não confere com o usuário\n")
            plt.imshow(cv2.cvtColor(imgDocumento,cv2.COLOR_BGR2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB))
            plt.show()
            

<img width="422" alt="image" src="https://github.com/luizhdramos/Fiap_Computer_Vision/assets/96795757/0247bd36-6f70-4963-89ec-1fcd333b34d0">

### Liviness Detection 

Foi realizado o treinamento de um modelo utilizando como base os pesos do modelo pre-treinado MobileNetV2, utilizando como input o `ImageDataGenerator` onde foi utilizado data augmentation nas amostras de treinamento. Com o modelo treinado ele foi consumido na função abaixo:


  
    def check_liviness(folder_path, filename):
         # Utiliza-se modelo pre-treinado de detecção de faces (Haar Cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image = cv2.imread(folder_path + filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cropped_face = image[y:y+h, x:x+w]
    
        cv2.imwrite(folder_path + 'face_' + filename , cropped_face)
    
        img = load_img(folder_path + 'face_' + filename, target_size=(224, 224))
        x = img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
    
        images = np.vstack([x])
        classes = model.predict(images, batch_size=32)
        print(classes[0])
    
        if classes[0]>0.5:
            print("\nA foto é real\n")
            plt.imshow(image_RGB)
            plt.show()
            plt.imshow(img)
            plt.show()
        else:
            print('\nA foto é fake\n')
            plt.imshow(image_RGB)
            plt.show()
            plt.imshow(img)
            plt.show()



<img width="405" alt="image" src="https://github.com/luizhdramos/Fiap_Computer_Vision/assets/96795757/43f95d05-d9db-40ba-9162-5e03197e9820">

## Nova versão e Melhoria no projeto

- Deixaria uma estação de trabalho com o ambiente virtual com as bibliotecas necessárias para o funcionamento basico do projeto
- Realizar um "Face Scrapping" em sites de video como o Youtube para aumentar a quantidade de amostras de fotos reais.
- Melhorar o processo de input de imagem, para ser obtido diretamente com uma webcam.
- Dentro dessa obtenção de imagem pela webcam, utilizaria o OpenCV para marcar o local onde o rosto do usuário deve ser posicionado, para geração de dados de maior qualidade




