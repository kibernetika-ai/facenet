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

Once you prepared facenet OpenVINO models and dataset, a classifier and
 placed them to the catalog, you are ready to start serving.

To start the serving, open catalog and find model **facenet-classifier**.
Next, choose the latest (or needed) version and click **Serve**. Then,
specify workspace and cluster and working directory **$SRC_DIR**.

Execution command:

```
kuberlab-serving --driver openvino --model-path $DATA_DIR/facenet.xml --hooks serving_hook.py -o classifier=/model/classifier.pkl -o flexible_batch_size=True -o resolutions=14x19,19x27,26x37,37x52,52x74,73x104,103x146,145x206,205x290 -o use_tf=true -o tf_path=$DATA_DIR
```

Explanation of execution command arguments:

* **--driver** openvino - use OpenVINO serving driver (as we serve facenet model in OpenVINO format)

* **--model-path** - model file location, in case of OpenVINO - **.xml** file location

* **--hooks <hook.py>** - special Python script containing pre/postprocess inference hooks,
they needed for applying intermediate neural networks such as PNet and ONet

* **-o classifier=<>** - Classifier file location

* **-o flexible_batch_size=true** - use any batch size for OpenVINO driver
(as we can see multiple faces on one picture)

* **-o resolutions=<>** - Use specified resolutions for PNet network

* **-o use_tf=true** - Use TensorFlow-based PNet/ONet networks (they are more
flexible and are working with any resolutions)

* **-o tf_path=<>** - Load TensoFlow-based model weights from specified location.

Next, specify requests and limits (at least 1-2 CPUs), image (CPU) -
`kuberlab/serving:latest-openvino` and **TCP** port 9000.

Then, need to add facenet model and source code volumes:

* Add this git repository as **src** with *subPath* **facenet/src**

* Add **facenet-pretrained** dataset with newly created version (named
 like **1.0.0-CPU-XXXXX**), it must include **facenet.xml**, **det{1-3}.npy** files,
 **pnet_*.xml** files and **onet.xml** file

Then, specify serving parameters for data input and output for the UI:

* output filter - **output**

* model - **any** (this value doesn't matter)

* out mime type - **image/png**

* click on checkbox **raw input**

Specify input params:

* name **input**, type **bytes**

