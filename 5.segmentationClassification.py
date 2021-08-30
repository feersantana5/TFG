import argparse
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh


# 1. Adquisicion de datos

def adquisicion_de_datos(dataset, clase_objeto, objeto, n_captura, n_frame):

    matrix_xyz = np.loadtxt(f"./{dataset}/xyzConf/{clase_objeto}/{objeto}_C{n_captura}_F{n_frame}.txt")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(matrix_xyz)
    print("Nube original: ")
    print(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    o3d.visualization.draw_geometries([pcd])

    return pcd


# 2. Calcular plano principal (pared o fondo) con RANSAC, segmentarlo, obtener la nube de outliers de ese plano

def segmentacion(pcd, umbral):

    plane_model, inliners = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Eliminado el plano : {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inliner_cloud = pcd.select_by_index(inliners)
    inliner_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = pcd.select_by_index(inliners, invert=True)

    print("Nube con plano segmentado a eliminar resaltado y resto de la nube: ")
    o3d.visualization.draw_geometries([inliner_cloud, outlier_cloud])

    print("Resto de la nube tras eliminar el plano: ")
    o3d.visualization.draw_geometries([outlier_cloud])  # puntos que no pertenecen

    # Eliminación de los puntos que quedan sueltos antes de clusterizar para mejorar el clusterizado y obtener el menor
    # numero de puntos no deseados

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Ruido a eliminar (rojo) y nube resultante (gris): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # elimina puntos que tienen menos del numero de vecinos indicados en la esfera "creada"
    cl, ind = outlier_cloud.remove_radius_outlier(nb_points=16, radius=0.05)

    display_inlier_outlier(outlier_cloud, ind)
    print("Nube resultante tras filtrar con el radio:")
    o3d.visualization.draw_geometries([cl])  # nube ya limpia

    # 3. Segmentacion (clusterizacion de la nube) de los puntos que no pertenecen al plano con DBSCAN()

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(cl.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
        max_label = labels.max() + 1
        print(f"La nube de puntos tiene: {max_label} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        print("Nube clusterizada: ")
        o3d.visualization.draw_geometries([cl])

        clusters = []
        boxes = []

        puntos_clusters_clasificados = []
        labels_clusters_clasificados = []


        # 3.2 Muestra Todos los clusters con su respectiva distancia, los que superan el umbral son posibles objetos y se envian para clasificar
        for i in range(max_label):
            id = np.where(labels == i)[0]
            pcd_i = cl.select_by_index(id)
            print(f"Cluster {i}: {pcd_i}")
            # resalto los cluster con una caja alineada roja
            aabb = pcd_i.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            clusters.append(pcd_i)
            boxes.append(aabb)
            if len(pcd_i.points) >= umbral:
                pcd_i_xyzarray = np.asarray(pcd_i.points)
                # calcula distancia a la que esta el objeto de profundidad
                print(f"El obstáculo está a {abs(np.max(pcd_i_xyzarray[:, 2]))} metros")
                # muestra el objeto
                o3d.visualization.draw_geometries([pcd_i])

                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_i, alpha=0.01)
                o3d.io.write_triangle_mesh(f"./pcd{i}.off", mesh)
                pcd_muestreada_puntos = trimesh.load(f"./pcd{i}.off").sample(2048)

                puntos_clusters_clasificados.append(pcd_muestreada_puntos)
                labels_clusters_clasificados.append(f"cluster{i}")

    # visualiza la nube limitada
    print("Todos los clusters delimitados en la nube filtrado")
    o3d.visualization.draw_geometries([clusters[i] for i in range(max_label)] + [boxes[i] for i in range(max_label)] + [cl])
    print("Clusters delimitados en la nube original")
    o3d.visualization.draw_geometries([clusters[i] for i in range(max_label)] + [boxes[i] for i in range(max_label)] + [cl] + [pcd])

    print(f"Hay {len(labels_clusters_clasificados)} clusters posibles de clasificar: {labels_clusters_clasificados}")

    return puntos_clusters_clasificados, labels_clusters_clasificados


# 4. Clasificacion
def clasificacion(num_points, clusters_clasificados, labels_clusters_clasificados, num_classes, batch_size, pesos, class_map):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    def get_model():
        def conv_bn(x, filters):
            x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
            x = layers.BatchNormalization(momentum=0.0)(x)
            return layers.Activation("relu")(x)

        def dense_bn(x, filters):
            x = layers.Dense(filters)(x)
            x = layers.BatchNormalization(momentum=0.0)(x)
            return layers.Activation("relu")(x)

        # PointNet consta de 2 componentes principales. La red MLP primaria y la red transformada(T-net)
        # La T-net tiene como objetivo aprender una matriz de transformacion afin mediante su propia mini red. Se usa 2 veces.
        # La primera vez para transformar las caracteristicas de entrada (n,3) en una representacion canonica.
        # La segunda, es una transformacion ain para la alineacion en el espacio de caracteristicas (n, 3).
        # La transformacion se restringe para estar cerca de una matriz ortogonal

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
            feat_T = layers.Reshape((num_features, num_features))(x)
            # Apply affine transformation to input features
            return layers.Dot(axes=(2, 1))([inputs, feat_T])

        # create the CNN (red neuronal convolucional)
        inputs = keras.Input(shape=(num_points, 3))  # Capa de entrada a traves de la clase de entrada, especifica la forma de muestra de entrada.

        x = tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)  # Dropout es un metodo de regularizacion inteligente que reduce el sobreajuste del dataset haciendo el modelo + robusto
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

        return model

    model = get_model()
    model.load_weights(pesos)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    def augment(points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label


    # 4.1 Prediccion para los clusters de la nube segmentada

    clasificacion_dataset = tf.data.Dataset.from_tensor_slices((clusters_clasificados, labels_clusters_clasificados))
    clasificacion_dataset = clasificacion_dataset.shuffle(len(clusters_clasificados)).map(augment).batch(batch_size)

    predictions = model.predict(clasificacion_dataset)

    for i in range(len(clusters_clasificados)):
        print(f"{labels_clusters_clasificados[i]} muestreado:")
        pcd = o3d.geometry.PointCloud()  # compruebo en open3d como quedo y la cantidad de puntos es correcta (2048)
        matrix = clusters_clasificados[i]
        pcd.points = o3d.utility.Vector3dVector(matrix)
        o3d.visualization.draw_geometries([pcd])
        print(pcd)

        score = tf.nn.softmax(predictions[i])
        print("El {} tiene mayor semejanza con la clase: {} con un {:.2f}% de confianza".format(labels_clusters_clasificados[i], class_map[np.argmax(score)], 100 * np.max(score)))

        plt.figure(figsize=(5, 5))
        plt.grid(False)
        plt.xticks(range(len(class_map.values())), class_map.values(), rotation=45)
        plt.yticks([])
        thisplot = plt.bar(range(num_classes), predictions[i], color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions[i])
        thisplot[predicted_label].set_color("red")
        plt.show()

def main():
    # valores por defecto
    num_points = 2048
    num_classes = 3
    batch_size = 24
    directorio = "MiDataCluster"
    clase_objeto = "silla"
    objeto = "SillaA"
    n_captura = 11
    n_frame = 0
    pesos = "./archivos/archivosTensor/weightsMiData3.h5"

    # diccionario con id y clase
    # class_map = {0: 'silla', 1: 'monitor'} # MiData2
    class_map = {0: 'silla', 1: 'mesa', 2: 'lampara'} # MiData3
    # class_map = {0: 'silla', 1: 'monitor', 2: 'mesa', 3: 'lampara'} # MiData4
    # class_map = {0: 'cama', 1: 'mesilla', 2: 'escritorio', 3: 'silla', 4: 'tocador', 5: 'mesa', 6: 'retrete', 7: 'sofa', 8: 'monitor', 9: 'banera'} #ModelNet


    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-d", "--dataset", type=str, default=directorio, help="ruta al directorio principal donde se encuentra la nube a segmentar")
    parser.add_argument("-c", "--clase_objeto", type=str, default=clase_objeto, help="clase del objeto a segmentar")
    parser.add_argument("-o", "--objeto", type=str, default=objeto, help="nombre del objeto a segmentar")
    parser.add_argument("-n", "--n_captura", type=int, default=n_captura, help="numero de captura a segmentar")
    parser.add_argument("-f", "--n_frame", type=int, default=n_frame, help="numero de captura a segmentar")
    parser.add_argument("-p", "--num_points", type=int, default=num_points, help="numero de puntos a muestrear la nube")
    parser.add_argument("-t", "--num_classes", type=int, default=num_classes, help="numero clases que tiene el problema")
    parser.add_argument("-b", "--batch_size", type=int, default=batch_size, help="tamaño del batch")
    parser.add_argument("-w", "--pesos", type=str, default=pesos, help="archivo que contiene los pesos")
    options = parser.parse_args()

    pcd = adquisicion_de_datos(options.dataset, options.clase_objeto, options.objeto, options.n_captura, options.n_frame)
    clusters_clasificados, labels_clusters_clasificados = segmentacion(pcd, options.num_points)
    clasificacion(options.num_points, clusters_clasificados, labels_clusters_clasificados, options.num_classes, options.batch_size, options.pesos, class_map)


if __name__ == "__main__":
    main()