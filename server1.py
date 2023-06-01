# from flask import Flask, request, jsonify
# import tensorflow as tf
# import cv2
# import base64
# import numpy as np

# app = Flask(__name__)
# model = tf.keras.models.load_model("mymodel.h5")

# @app.route('/upload', methods=['POST'])

# def classify_flower(image):
#     # Code xử lý ảnh và nhận biết loại hoa ở đây
#     # ...

#     # Giả sử kết quả là flower_name và accuracy
#     flower_name = ['pink' , 'orchid', 'bell' , 'bean' , 'marigold']
#     accuracy = [1,2,3,4,5]

#     return flower_name, accuracy

# def upload():
#     file = request.files['image']
#     file.save('received_image.jpg')
    
#     # Đưa ảnh vào model để nhận biết loại hoa
#     img = cv2.imread('received_image.jpg')
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0
#     img = tf.expand_dims(img, axis=0)
#     predictions = model.predict(img)
#     flower_index = tf.argmax(predictions, axis=1).numpy()[0]
#     # Mappings giữa index và tên loại hoa
#     flower_names = ['pink' , 'orchid', 'bell' , 'bean' , 'marigold']
#     # Tên loại hoa được nhận biết
#     flower_name = flower_names[flower_index]
#     return jsonify({'flower_name': flower_name})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import base64
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("model5.h5")

def classify_flower(image):
    img = cv2.resize(image, (150, 150))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    predictions = model.predict(img)
    flower_index = tf.argmax(predictions, axis=1).numpy()[0]
    # flower_names = ['bean', 'bell', 'firelily', 'lotus', 'marigold' , 'orchid' , 'pink', 'rose' ,'tulip']
    flower_names = ['barbeton daisy','bell','cape flower','firelily','fritillary','great masterwort','lotus','marigold','orchid','osteospermum','pink-yellow dahlia','primula','purple coneflower','rose','sweet william','thorn apple','trumpet creeper','wallflower','watercress','waterlily']
    flower_name = flower_names[flower_index]
    accuracy = predictions[0][flower_index]
    return flower_name, accuracy

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    # image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)


    flower_name, accuracy = classify_flower(image)

    return jsonify({'flower_name': flower_name, 'accuracy': float(accuracy)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
