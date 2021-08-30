import argparse
import os
from sklearn.metrics import classification_report
import glob
import trimesh
import numpy as np
import open3d as o3d
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# configuracion para la memoria de la jetson
def config_hardware():
    device = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
    tf.config.experimental.set_virtual_device_configuration(device[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# 0. PREPARAR DATASET

# funcion para parsear los datos: cada nube o malla es convertida en numpy array
def parse_dataset(data_dir, num_points, class_map):
    train_points = []  # puntos de entrenamientos
    train_labels = []  # etiquetas de entrenamiento
    test_points = []  # puntos de testeo
    test_labels = []  # etiquetas de testeo

    folders = glob.glob(os.path.join(data_dir, "[!README]*"))  # directorio de las carpetas
    valores = list(class_map.values())

    for i, folder in enumerate(folders):
        print("Procesando clase: {}".format(os.path.basename(folder)))
        # verifica valor de los valores
        key = valores.index(folder.split("/")[-1])
        # obtenemos los archivos de cada carpeta
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            # Preprocesado con open3d

            # Para midata (.ply)
            pcd = o3d.io.read_point_cloud(f)
            # Para midata (.txt)
            # pcd = o3d.geometry.PointCloud()
            # matrix_xyz = np.loadtxt(f)
            # pcd.points = o3d.utility.Vector3dVector(matrix_xyz)

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.01)
            filename = os.path.basename(f.split(".")[1])
            o3d.io.write_triangle_mesh(f"./MiDataCluster/clustersConf/mesh/train/{filename}.off", mesh)
            pcd_muestreada_train_puntos = trimesh.load(f"./MiDataCluster/clustersConf/mesh/train/{filename}.off").sample(num_points)

            # Para modelnet (mallas.off))
            # pcd_muestreada_train_puntos = trimesh.load(f).sample(num_points)  # muestrea la malla con N puntos

            train_points.append(pcd_muestreada_train_puntos)

            train_labels.append(key)

        for f in test_files:
            # Preprocesado con open3d

            # Para midata (.ply)
            pcd = o3d.io.read_point_cloud(f)
            # Para midata (.txt)
            # pcd = o3d.geometry.PointCloud()
            # matrix_xyz = np.loadtxt(f)
            # pcd.points = o3d.utility.Vector3dVector(matrix_xyz)

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.01)
            filename = os.path.basename(f.split(".")[1])
            o3d.io.write_triangle_mesh(f"./MiDataCluster/clustersConf/mesh/test/{filename}.off", mesh)
            pcd_muestreada_test_puntos = trimesh.load(f"./MiDataCluster/clustersConf/mesh/test/{filename}.off").sample(num_points)

            # Para modelnet (mallas.off))
            # pcd_muestreada_test_puntos = trimesh.load(f).sample(num_points)  # muestrea la malla con N puntos

            test_points.append(pcd_muestreada_test_puntos)

            test_labels.append(key)

    print(class_map)
    print(train_labels)
    print(test_labels)

    print(len(train_points))
    print(len(train_labels))

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels)
    )


def comprobar_dataset(train_points, train_labels, class_map):
    # Verificar que el conjuntos de datos se ve correctamente, muestro las primeras 15 nubes de conjunto de
    # entrenamiento
    fig = plt.figure(figsize=(10, 10))
    for i in range(15):
        ax = fig.add_subplot(3, 5, i + 1, projection="3d")
        ax.scatter(train_points[i, :, 0], train_points[i, :, 1], train_points[i, :, 2], color='green')
        ax.set_title("label: {:}".format(class_map[train_labels[i]]))
        ax.set_axis_off()
    plt.show()


def prepare_dataset(train_points, test_points, train_labels, test_labels, batch_size):
    # augmentation for train dataset
    # agita y mezclar el conjunto de datos de entrenamiento

    def augment(points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

    return train_dataset, test_dataset


def create_model(num_points, num_classes, results):

    # 1. BUILD MODEL
    print("1. BUILD MODEL")

    # Cada convolucion y capa consiste de: Convolucion / Densa --> Batch normalization --> ReLU Activation

    # capa convolcional
    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    # capa densa
    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    # PointNet consta de 2 componentes principales. La red MLP primaria y la red transformada(T-net) La T-net tiene
    # como objetivo aprender una matriz de transformacion afin mediante su propia mini red. Se usa 2 veces. La
    # primera vez para transformar las caracteristicas de entrada (n,3) en una representacion canonica. La segunda,
    # es una transformacion ain para la alineacion en el espacio de caracteristicas (n, 3). La transformacion se
    # restringe para estar cerca de una matriz ortogonal

    class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg=0.001):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    # function to create t-net layers

    def tnet(inputs, num_features):
        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(num_features * num_features, kernel_initializer="zeros", bias_initializer=bias, activity_regularizer=reg, )(x)
        feat_t = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_t])

    # create the CNN (red neuronal convolucional)
    inputs = keras.Input(shape=(num_points, 3))  # Capa de entrada a traves de la clase de entrada, especifica la forma de muestra de entrada.

    x = tnet(inputs, 3)  # Transforma entrada
    # MLP (2 capas de 32 neuronas c/u)
    x = conv_bn(x, 32)  # Extrae caracteristicas
    x = conv_bn(x, 32)  # Extrae caracteristicas
    x = tnet(x, 32)  # Transforma caracteristicas de mtrix 32 x 32
    # MLP (3 capas de 32, 64 y 512 neuronas)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    # dota de invariancia a permutaciones
    x = layers.GlobalMaxPooling1D()(x)
    # MLP (3 capas fully connected 256, 128, k)
    x = dense_bn(x, 256)  # Max pooling + dense para hacer la clasificacion
    x = layers.Dropout(0.3)(x)  # Dropout es un metodo de regularizacion inteligente que reduce el sobreajuste del dataset haciendo el modelo + robusto
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)  # la salida esta condicionada al numero de clases, dado que tiene ese num de neuronas, softmax para obtener la prediccion

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet") # especificar neuronas de entradas, salida y nombre
    model.summary()  # descripcion del modelo: resumen de cada capa por consecuente total del sistema. permte ver las formas de salida y numero de parametros(pesos) del modelo
    plot_model(model, f".{results}arquitecturaModelo.png", show_shapes=True)  # crea grafico del modelo en cuado para c/capa medinate flechas y flujo de datos

    return model

def compile_model(model):
    # 2. COMPILE MODEL
    print("2.COMPILE")

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )


def train_model(model, train_dataset, test_dataset, epochs, peso):
    # 3. TRAIN MODEL
    print("3.TRAIN")
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # GUARDAR PESOS
    model.save_weights(peso)

    return history


def evaluate_model(model, test_points, test_labels, history, class_map,  num_classes, epochs, results):
    # 4. EVALUATE MODEL
    print("4.EVALUATE")
    test_loss, test_acc = model.evaluate(test_points, test_labels, verbose=1)

    print("Test Accuracy: %.2f" % test_acc)
    print("Test Loss: %.2f" % test_loss)

    # all data in history
    # print(history.history.keys())

    print("[INFO] Generando gráficos de evaluación del modelo...")

    # summarize history for accuracy & loss

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(range(epochs), history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history.history['loss'], label='Training Loss')
    plt.plot(range(epochs), history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(f".{results}accuracy&loss.png")
    plt.show()

    # summarize history for accuracy

    plt.figure("Accuracy")
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch Nº')
    plt.grid()
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(f".{results}accuracy.png")
    plt.show()

    # summarize history for loss
    plt.figure("Loss")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch Nº')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(f".{results}loss.png")
    plt.show()

    # evaluar el modelo con el conjunto de entrenamiento mediante la matriz de confusión

    pred_todo = model.predict(test_points)
    preds_todo = tf.math.argmax(pred_todo, -1)

    # matriz confusion skcit-learn
    # sns.heatmap(confusion_matrix(labels.numpy(), preds.numpy()), annot=True)

    matriz_confusion = tf.math.confusion_matrix(labels=test_labels, predictions=preds_todo.numpy(), num_classes=num_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion, xticklabels=np.array(list(class_map.values())), yticklabels=np.array(list(class_map.values())), annot=True, fmt="g")
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.savefig(f".{results}matriz_confusionTodo.png")
    plt.show()

    # classification report con el conjunto de entrenamiento

    report = classification_report(test_labels, preds_todo.numpy(), target_names=np.array(list(class_map.values())))
    print(report)

    return report, test_acc


def predict(test_dataset, model, class_map, batch_size, num_points, num_classes, results):
    # 5. VISUALIZE PREDICTIONS

    # 5.1 Prediccion para 10 nubes del dataset de testeo
    data = test_dataset.take(10)
    points, labels = list(data)[0]

    points = points[:10, ...]
    labels = labels[:10, ...]

    # run test data through model
    predictions = model.predict(points)
    preds = tf.math.argmax(predictions, -1)

    points = points.numpy()

    # plot points with predicted class and label
    fig = plt.figure(figsize=(15, 10))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], color='red')
        ax.set_title("pred: {:}, label: {:}".format(class_map[preds[i].numpy()], class_map[labels.numpy()[i]]))
        ax.set_axis_off()
    plt.savefig(f".{results}prediction.png", bbox_inches='tight')
    plt.show()


    # 5.2 Prediccion para 1 nube externa, dato nuevo (silla)

    pcd = o3d.io.read_point_cloud("./archivos/sillaC_1.ply")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.01)
    o3d.io.write_triangle_mesh(f"./prediction.off", mesh)

    pcd_muestreada_puntos_prediction = trimesh.load(f"./prediction.off").sample(num_points)

    pp_points = []
    pp_labels = []
    pp_points.append(pcd_muestreada_puntos_prediction)
    pp_labels.append("pcd_muestreada_puntos_prediction")

    def augment(points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label

    pp_dataset = tf.data.Dataset.from_tensor_slices((pp_points, pp_labels))
    pp_dataset = pp_dataset.shuffle(len(pp_points)).map(augment).batch(batch_size)

    predictions = model.predict(pp_dataset)
    score = tf.nn.softmax(predictions[0])
    print("La nube tiene mayor semejanza con la clase: {} con un {:.2f}% de confianza".format(class_map[np.argmax(score)], 100 * np.max(score)))

    # Hacer graficas
    def plot_value_array(predictions_array):
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color("red")

    plt.figure(figsize=(15, 10))
    plot_value_array(predictions[0])
    _ = plt.xticks(range(num_classes), class_map.values(), rotation=45)
    plt.savefig(f".{results}predictionExterna.png", bbox_inches='tight')
    plt.show()


def save_results(history, result_dir, num_points, num_classes, batch_size, report, test_acc):
    loss = history.history['loss']
    acc = history.history['sparse_categorical_accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_sparse_categorical_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write("Parámetros del entrenamiento:\n\n")
        fp.write(f"Numero de puntos de muestreo: {num_points}\n")
        fp.write(f"Numero de clases: {num_classes}\n")
        fp.write(f"Batch size: {batch_size}\n")
        fp.write(f"Numero de epochs: {len(acc)}\n\n")

        fp.write('\n\nResultados del entrenamiento:\n\n')

        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

        fp.write(f"\n\nPrecisión de Test: {test_acc * 100} %")

        fp.write('\n\nClassification Report\n\n{}\n\n'.format(report))
        fp.close()


def main():

    # valores por defecto
    num_points = 2048
    data_dir = "MiDataClusterPrep3"
    num_classes = 3
    batch_size = 24
    epochs = 240
    results = "/archivos/archivosTensor/MiData3/"
    pesos = "./archivos/archivosTensor/MiData3/weightsMiDataCluster3.h5"


    # diccionario con id y clase
    # class_map = {0: 'silla', 1: 'monitor'}  # MiData2
    class_map = {0: 'silla', 1: 'mesa', 2: 'lampara'}  # MiData3
    # class_map = {0: 'silla', 1: 'monitor', 2: 'mesa', 3: 'lampara'}  # MiData4
    # class_map = {0: 'cama', 1: 'mesilla', 2: 'escritorio', 3: 'silla', 4: 'tocador', 5: 'mesa', 6: 'retrete', 7: 'sofa', 8: 'monitor', 9: 'banera'}  # ModelNet

    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-d", "--data_dir", type=str, default=data_dir, help="ruta al directorio principal donde se encuentra el conjunto de datos")
    parser.add_argument("-p", "--num_points", type=int, default=num_points, help="numero de puntos a muestrear la nube")
    parser.add_argument("-t", "--num_classes", type=int, default=num_classes, help="numero clases que tiene el problema")
    parser.add_argument("-b", "--batch_size", type=int, default=batch_size, help="tamaño del batch")
    parser.add_argument("-e", "--epochs", type=int, default=epochs, help="numero de epochs")
    parser.add_argument("-w", "--pesos", type=str, default=pesos, help="archivo donde guardar los pesos, el formato debe ser .h5")
    parser.add_argument("-r", "--results", type=str, default=results, help="directorio para guardar los resultados")
    options = parser.parse_args()


    config_hardware()  # SOLO EN JETSON
    train_points, test_points, train_labels, test_labels = parse_dataset(options.data_dir, options.num_points, class_map)
    comprobar_dataset(train_points, train_labels, class_map)
    train_dataset, test_dataset = prepare_dataset(train_points, test_points, train_labels, test_labels, options.batch_size)
    model = create_model(options.num_points, options.num_classes, options.results)
    compile_model(model)
    history = train_model(model, train_dataset, test_dataset, options.epochs, options.pesos)
    report, test_acc = evaluate_model(model, test_points, test_labels, history, class_map, options.num_classes, options.epochs, options.results)
    predict(test_dataset, model, class_map, options.batch_size, options.num_points, options.num_classes, options.results)
    save_results(history, f"{dir_path}{results}", options.num_points, options.num_classes, options.batch_size, report, test_acc)


if __name__ == "__main__":
    main()