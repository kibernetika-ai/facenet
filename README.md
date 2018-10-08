# Face Detection and Recognition using OpenVino

### Facenet on OpenVINO
Pipeline uses pre-trained model **facenet-pretrained**

Training pipeline consists of several tasks responsible for data preparation, model training, and validation.
The resulting model will be placed to the catalog ready for deployment to outside devices.

#### Task list in the project:

* **align-images:** Task which looks in the specified folder for faces
classes and prepares them to be recognized.

* **train-classifier:** Task iterates over prepared images from
align-images and computes embeddings of each face. Then it trains and
 saves a classifier with connections to the face classes. The task can
  be executed using either tensorflow or OpenVINO model.

* **validate-classifier:** Task loads the classifier and checks every
 embedding computed from the images from the dataset. It ensures that
  each embedding belongs to the corresponding person. After validation,
   the new model is versioned and placed in the catalog.

* **pipeline:** The aggregate task which will execute pipeline workflow:
 align, train and validate.

* **model-converter:** Task which converts all models needed for Facenet
 from Tensorflow to OpenVINO format.


### Start facenet serving

Once you prepared facenet OpenVINO models, a classifier and placed them
to the catalog, you are ready to start serving.
